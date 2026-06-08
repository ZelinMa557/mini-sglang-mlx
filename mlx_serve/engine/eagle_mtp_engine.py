"""EAGLE-style MTP (multi-token prediction) speculative-decoding engine.

This engine drives the Qwen3.5 MTP draft model alongside the target.
Each decode iter does:

    * ``K-1`` regular draft forwards (``K = num_mtp_step``) to produce
      drafts ``D_2..D_K``.  ``D_1`` itself comes from the previous
      iter / prefill — so ``K-1`` forwards yield ``K`` total drafts.
      Each forward writes the draft K/V at its position using the
      draft's own previous hidden as a *proxy* for the target hidden.
    * A single target verify pass over ``[T, D_1, ..., D_K]`` (K+1
      tokens) that greedy-accepts the longest matching prefix and
      yields real target hidden states ``h_c..h_{c+K}``.
    * A *calibration prefill* on the draft model over
      ``[D_1, .., D_{j}, bonus]`` per req, feeding the verify-time
      target hidden as ``target_hidden_states``.  This overwrites the
      proxy-hidden K/V from the regular forwards with K/V derived
      from REAL target hidden, keeping draft accuracy from drifting
      over long generations, AND produces ``pending_draft_token`` /
      ``pending_draft_hidden`` for the next iter (last sample / last
      hidden of the calibration).

Total draft forwards per decode iter: ``K - 1`` decode steps + 1
prefill (sized ``j+1 ≤ K+1`` tokens per req).

Mirrored page layout
--------------------
Target and draft KV caches share the SAME page IDs at the same logical
positions — :class:`AttnBackend` of the draft is wired to the target's
``page_table``, and the draft :class:`MHAKVCache` has the same
``num_pages`` as the target.  This means:

* No separate draft page allocator / page table.
* Every page allocated by the target's :class:`CacheManager` is also
  the slot index for the draft to write its K/V into.
* Prefix cache "just works" — when a finished request's pages are
  inserted into the radix, the same page IDs hold both target and
  draft K/V, ready for the next request to reuse.

Conventions (all per-req):
    ``cached_len`` = number of tokens whose KV is committed to the
        target cache (target's logical sequence length so far). Draft
        KV length is identical to this at every iter boundary.
    ``pending_token`` = next token sampled but not yet in target KV;
        sits at ``input_ids[cached_len]`` and is the first verify
        position next iter.
    ``pending_draft_token`` / ``pending_draft_hidden`` = first draft
        for next iter and the draft model's hidden state used as
        ``target_hidden_states`` proxy for the first regular draft step.

Limitations (intentional first cut):
    * Greedy verification only (temperature 0).
    * No chunked prefill.
    * Draft model must be single-layer full-attention (Qwen3.5 MTP).
    * Prefix cache: under the shifted-draft convention, the cached
      draft K/V at the *boundary* position ``matched_len - 1`` depends
      on the original request's ``input_ids[matched_len]``, which may
      not equal the new request's.  We accept this mismatch — target
      correctness is unaffected (target verifies every token), but the
      draft acceptance rate may dip slightly right after a prefix hit.
"""

from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx

from mlx_serve.attention import AttnBackend
from mlx_serve.core import Batch, BatchPhase, Context, Req, use_ctx
from mlx_serve.kvcache.mha_pool import MHAKVCache
from mlx_serve.models.qwen3_5_mtp import load_qwen3_5_mtp_draft_model
from mlx_serve.utils import init_logger

from .config import EngineConfig
from .spec_engine import SpecEngine, SpecForwardOutput
from .spec_sample import GreedyVerifyResult, greedy_verify

logger = init_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# Engine
# ════════════════════════════════════════════════════════════════════


class EagleMTPEngine(SpecEngine):
    """EAGLE-style MTP engine over :class:`SpecEngine`.

    Shared infrastructure (target model + KV / Mamba pools, per-iter
    slab allocation + verify metadata builders + resource release)
    lives on the base class.  This subclass additionally owns:

    * the Qwen3.5 MTP single-layer draft model (loaded here, shares
      ``embed_tokens`` / ``lm_head`` with the target).
    * a 1-layer MHA KV cache for the draft, with the SAME ``num_pages``
      as the target so page IDs are usable interchangeably between
      both KV buffers (mirrored layout).
    * a separate ``Context`` for the draft (so model layers picking up
      ``get_global_ctx().attn_backend`` see the draft backend), but
      sharing the target's ``page_table`` — the target's
      :class:`CacheManager` is the sole source of truth for page
      allocation and freeing.
    """

    def __init__(self, config: EngineConfig):
        assert config.mtp_model_path is not None, (
            "EagleMTPEngine requires EngineConfig.mtp_model_path"
        )
        assert config.num_mtp_step >= 1, (
            f"num_mtp_step must be >= 1, got {config.num_mtp_step}"
        )
        super().__init__(config)
        self.K = config.num_mtp_step

        # Load the MTP draft (shares embed_tokens / lm_head with target).
        logger.info("Loading MTP draft model from %s", config.mtp_model_path)
        self.draft_model, _ = load_qwen3_5_mtp_draft_model(
            config.mtp_model_path, self.model
        )
        logger.info("MTP draft model loaded and shared embed/lm_head.")

        # Draft KV cache: mirrors the target's page layout.  Same
        # ``num_pages`` so any target-side page ID is also a valid
        # slot in the draft buffer; per-token shape matches the
        # target's full-attention layers, but only 1 layer (Qwen3.5
        # MTP module).  Total bytes is ~num_target_layers smaller
        # than the target KV cache, so this is cheap.
        self.draft_kv_cache = MHAKVCache(
            num_kv_heads=self.model_meta.num_kv_heads,
            num_layers=1,
            head_dim=self.model_meta.head_dim,
            num_pages=self.num_pages + 1,  # +1 mirrors target's dummy slot
            dtype=self.dtype,
        )

        # Reuse the target's page_table — every page ID written by
        # the CacheManager (radix or naive) is simultaneously the
        # slot index into both target_kv_cache and draft_kv_cache.
        self.draft_attn_backend = AttnBackend(
            config=self.model_meta,  # type: ignore[arg-type]
            kvcache=self.draft_kv_cache,
            page_table=self.page_table,
        )
        self.draft_ctx = Context(
            page_size=1,
            attn_backend=self.draft_attn_backend,
            mamba_pool=None,
            gdn_backend=None,
        )

        # The MTP draft consumes the target's LAST hidden as
        # ``target_hidden_states`` (the same tensor that goes into
        # the target's lm_head).  We capture it via the post-norm
        # special-case of :meth:`Qwen3_5Model.__call__`.
        self._target_capture_layer_ids: Tuple[int, ...] = (
            len(self.model.layers) - 1,
        )

    def _extra_mamba_checkpoints_per_req(self, config: EngineConfig) -> int:
        # Target verify checkpoints ``K`` extra states per req.
        return config.num_mtp_step

    # ════════════════════════════════════════════════════════════════
    # Prefill: target prefill → sample T_1 → shifted draft prefill
    # ════════════════════════════════════════════════════════════════

    def _run_prefill(self, batch: Batch) -> SpecForwardOutput:
        from mlx_serve.scheduler.prefill import ChunkedReq  # local: avoid cycle

        reqs = batch.reqs
        B = len(reqs)
        # Chunked prefill is unsupported in MTP: the draft prefill needs
        # target hidden states at every prompt position, so the target
        # must process the full prompt in one shot.
        for req in reqs:
            if isinstance(req, ChunkedReq):
                raise NotImplementedError(
                    "Chunked prefill is not supported by EagleMTPEngine; "
                    "increase ``max_extend_tokens`` to fit the longest "
                    "prompt or shorten inputs."
                )

        # 1. Target prefill (this also writes the target page_table /
        # mamba metadata, identical to scheduler._prepare_batch).  With
        # prefix cache, ``extend_len = N - cached_len``; target re-runs
        # only the NEW positions and we get target_hidden for those.
        self._prepare_target_prefill_inplace(batch)
        with self.ctx.forward_batch(batch):
            (target_hidden,), target_logits = self.model(
                capture_layer_ids=self._target_capture_layer_ids,
            )

        # 2. Sample T_1 per req at the last prompt position.
        last_indices = batch.attn_metadata.get_last_indices(B)
        T_batch = mx.argmax(target_logits[last_indices], axis=-1).astype(mx.int32)

        # 3. Draft prefill — process only the new positions
        # ``cached_len..device_len-1`` (length = extend_len per req).
        # Inputs are shifted: at draft position p we feed input_ids[p+1]
        # (or T_1 for the last position) plus target_hidden[p].
        #
        # Page IDs are MIRRORED with the target: we reuse
        # ``batch.out_loc`` directly so draft writes its K/V to the same
        # page IDs that target just used for the extend positions.
        # Cached positions 0..cached_len-1 of the draft already hold
        # valid K/V (written by whichever earlier MTP iter produced the
        # prefix entry, target+draft together).
        draft_input_ids = self._build_draft_prefill_input_ids(reqs, T_batch)
        draft_batch = self._prepare_draft_prefill_batch(reqs, batch.out_loc)
        with use_ctx(self.draft_ctx), self.draft_ctx.forward_batch(draft_batch):
            draft_hidden, draft_logits = self.draft_model(
                draft_input_ids, target_hidden, return_hidden=True
            )

        # 4. Pull out per-req last position: dh_{N-1} and D_1.
        draft_last_indices = draft_batch.attn_metadata.get_last_indices(B)
        last_dh = draft_hidden[draft_last_indices]  # [B, D]
        D1_batch = mx.argmax(
            draft_logits[draft_last_indices], axis=-1
        ).astype(mx.int32)  # [B]

        # Realise outputs once; per-req slicing below is just views.
        mx.eval(T_batch, D1_batch, last_dh)

        # 5. Commit T_1 to req state (mirrors what a non-MTP forward
        # does via ``req.complete_one()`` + ``append_host``).  We commit
        # ``extend_len`` tokens to the target KV and append the freshly
        # sampled T_1, leaving ``input_ids[cached_len] = T_1`` as the
        # pending token for the first decode iter.
        accepted: List[mx.array] = [T_batch[i : i + 1] for i in range(B)]
        for i, req in enumerate(reqs):
            n_committed = req.extend_len
            req.append_host(accepted[i])
            req.complete_many(n_committed)
            req.pending_token = accepted[i]
            req.pending_draft_token = D1_batch[i : i + 1]
            req.pending_draft_hidden = last_dh[i]
        return SpecForwardOutput(accepted_tokens=accepted)

    # ════════════════════════════════════════════════════════════════
    # Decode iter: K-1 draft steps → verify → calibration prefill
    # ════════════════════════════════════════════════════════════════

    def _run_decode_iter(self, batch: Batch) -> SpecForwardOutput:
        K = self.K
        reqs = batch.reqs
        B = len(reqs)
        verify_len = K + 1

        # ---- Pre-allocate B*(K+1) shared pages -----------------------
        # One slab per req, K+1 pages wide, covering positions
        # cached_len..cached_len+K.  The SAME page IDs are used by both
        # target verify (K+1 positions) and the draft work this iter
        # (K-1 regular forwards write slabs[:, 0..K-2]; calibration
        # prefill writes slabs[:, 0..j_i] where j_i ≤ K).  Slot i of
        # slab[r] simultaneously addresses target_kv_cache and
        # draft_kv_cache via the mirrored page layout.
        flat_pages, slabs = self._allocate_iter_slabs(reqs)

        # ---- Gather pending state from prev iter ---------------------
        # ``stack(...).squeeze(-1)`` collapses a list of [1]-shape
        # tensors into a [B] tensor; pending_draft_hidden is [D] each.
        pending_T = mx.stack([r.pending_token for r in reqs]).squeeze(-1)
        cur_token = mx.stack([r.pending_draft_token for r in reqs]).squeeze(-1)
        cur_hidden = mx.stack([r.pending_draft_hidden for r in reqs])

        # ---- K-1 regular draft decode forwards -----------------------
        # ``D_1 = pending_draft_token`` came from the prev iter / prefill,
        # so we only need K-1 more forwards (D_2..D_K) to hand the target
        # K drafts to verify.  Step ``k`` writes draft K/V at page ID
        # ``slabs[:, k]`` using the draft's own previous hidden as a
        # *proxy* for target hidden — the calibration prefill below
        # overwrites this with K/V derived from real target hidden.
        #
        # When ``K == 1`` this loop is empty: ``drafts = [D_1]`` already.
        drafts_per_step: List[mx.array] = [cur_token]
        for step in range(K - 1):
            cur_lens = [r.cached_len + step for r in reqs]
            step_out_loc = slabs[:, step]  # [B] page IDs
            draft_batch = self._build_draft_decode_batch(
                reqs, cur_lens, step_out_loc,
            )
            with (
                use_ctx(self.draft_ctx),
                self.draft_ctx.forward_batch(draft_batch),
            ):
                cur_hidden, d_logits = self.draft_model(
                    cur_token, cur_hidden, return_hidden=True
                )
            cur_token = mx.argmax(d_logits, axis=-1).astype(mx.int32)
            drafts_per_step.append(cur_token)
        drafts = mx.stack(drafts_per_step, axis=1)  # [B, K]

        # ---- Target verify on [T, D_1, ..., D_K] --------------------
        verify_input_ids = mx.concatenate(
            [pending_T[:, None], drafts], axis=1
        ).reshape(-1)  # [B*(K+1)]
        verify_batch, slot_ids_rows, new_mamba_slots = (
            self._build_target_verify_batch(
                reqs, verify_input_ids, flat_pages,
            )
        )
        with self.ctx.forward_batch(verify_batch):
            (verify_hidden_flat,), verify_logits_flat = self.model(
                capture_layer_ids=self._target_capture_layer_ids,
            )
        D = verify_hidden_flat.shape[-1]
        V = verify_logits_flat.shape[-1]
        verify_hidden = verify_hidden_flat.reshape(B, verify_len, D)
        verify_logits = verify_logits_flat.reshape(B, verify_len, V)

        # ---- Greedy acceptance + single host sync -------------------
        # Counts + bonus + drafts in one eval so the subsequent per-req
        # bookkeeping doesn't trigger extra flushes.  ``verify_hidden``
        # is left lazy: the calibration prefill below consumes slices
        # of it on-device.
        result: GreedyVerifyResult = greedy_verify(verify_logits, drafts)
        mx.eval(result.num_drafts_accepted, result.bonus_tokens, drafts)
        num_accepted_host: List[int] = result.num_drafts_accepted.tolist()
        bonus_host: List[int] = result.bonus_tokens.tolist()
        drafts_host: List[List[int]] = drafts.tolist()

        # ---- Draft calibration prefill ------------------------------
        # Re-run draft over (D_1..D_{j_i}, bonus_i) per req with REAL
        # verify hidden as ``target_hidden_states``, overwriting the
        # proxy-hidden K/V at every newly-committed position with K/V
        # derived from real target hidden.  This keeps the draft K/V
        # cache "calibrated" so its accuracy doesn't drift across long
        # generations.  Also produces ``pending_draft_*`` for the next
        # iter (last logits / hidden of this prefill).
        calib_batch, calib_input_ids, calib_target_hidden = (
            self._build_draft_calibration_batch(
                reqs, drafts, result.bonus_tokens, verify_hidden,
                slabs, num_accepted_host,
            )
        )
        with (
            use_ctx(self.draft_ctx),
            self.draft_ctx.forward_batch(calib_batch),
        ):
            new_dh, new_dlogits = self.draft_model(
                calib_input_ids, calib_target_hidden, return_hidden=True,
            )
        last_indices = calib_batch.attn_metadata.get_last_indices(B)
        new_dh_last = new_dh[last_indices]  # [B, D]
        new_D1 = mx.argmax(
            new_dlogits[last_indices], axis=-1,
        ).astype(mx.int32)  # [B]
        mx.eval(new_D1, new_dh_last)

        # ---- Release per-iter resources -----------------------------
        # Strictly AFTER the calibration prefill so that any pages /
        # mamba state the calibration needed to read are still live.
        self._release_iter_resources(
            reqs, num_accepted_host, slabs, slot_ids_rows, new_mamba_slots,
        )

        # ---- Per-req state update + accepted_tokens output ----------
        accepted: List[mx.array] = []
        for i, r in enumerate(reqs):
            j = num_accepted_host[i]
            # New tokens this iter that go on input_ids: D_1..D_j + bonus.
            tail_list = drafts_host[i][:j] + [bonus_host[i]]
            tail = mx.array(tail_list, dtype=mx.int32)
            r.input_ids = mx.concatenate([r.input_ids, tail])
            r.complete_many(j + 1)
            r.pending_token = tail[-1:]
            r.pending_draft_token = new_D1[i : i + 1]
            r.pending_draft_hidden = new_dh_last[i]
            accepted.append(tail)
        return SpecForwardOutput(accepted_tokens=accepted)

    # ════════════════════════════════════════════════════════════════
    # Batch / metadata builders (MTP-specific; shared helpers live
    # in :class:`SpecEngine`).
    # ════════════════════════════════════════════════════════════════

    def _build_draft_prefill_input_ids(
        self, reqs: List[Req], T_batch: mx.array,
    ) -> mx.array:
        """Per-req shifted draft inputs over the EXTEND range only.

        For each req with cached_len=c, device_len=N, T_i sampled by
        target: produces ``[y_{c+1}, y_{c+2}, ..., y_{N-1}, T_i]``
        (length = extend_len = N - c).  Concatenated in batch order so
        each contiguous slab aligns with the matching rows of
        ``target_hidden`` returned by target prefill.
        """
        parts: List[mx.array] = []
        for i, req in enumerate(reqs):
            shifted = req.input_ids[req.cached_len + 1 : req.device_len]
            parts.append(mx.concatenate([shifted, T_batch[i : i + 1]]))
        return mx.concatenate(parts)

    def _prepare_draft_prefill_batch(
        self, reqs: List[Req], target_out_loc: mx.array,
    ) -> Batch:
        """Build draft prefill metadata that mirrors the target batch.

        We reuse ``target_out_loc`` directly — the page IDs the target's
        :class:`CacheManager` just handed out for the extend positions
        are also the slots the draft writes its K/V into.  The
        ``page_table`` was already populated by
        :meth:`_prepare_target_prefill_inplace`; we simply ask the
        draft's :class:`AttnBackend` to build prefill metadata over the
        same ``cached_len..device_len`` window (default behaviour).

        The model takes ``input_ids`` explicitly via ``__call__`` so
        ``batch.input_ids`` here is a placeholder.
        """
        metadata = self.draft_attn_backend.build_prefill_metadata(reqs)
        batch = Batch(reqs=reqs, phase=BatchPhase.PREFILL)
        batch.input_ids = mx.array([], dtype=mx.int32)
        batch.out_loc = target_out_loc  # mirror the target's page IDs
        batch.padded_reqs = reqs
        batch.attn_metadata = metadata
        return batch

    def _build_draft_decode_batch(
        self,
        reqs: List[Req],
        draft_kv_lens: List[int],
        out_loc: mx.array,
    ) -> Batch:
        """Single draft decode step (1 new token per req).

        ``draft_kv_lens[i]`` is the draft cache length BEFORE this step
        (also the RoPE position of the new token).  ``out_loc`` is the
        per-req page IDs to write the new K/V into ([B] int32); the
        caller has already stamped ``page_table[r, draft_kv_lens[i]] =
        out_loc[i]`` (via the shared slab) so the AttnBackend's
        metadata builder just reads back from ``page_table``.
        """
        kv_lens_after = [l + 1 for l in draft_kv_lens]
        positions = mx.array(draft_kv_lens, dtype=mx.int32)
        metadata = self.draft_attn_backend.build_decode_metadata(
            reqs, kv_lens=kv_lens_after, positions=positions,
        )

        batch = Batch(reqs=reqs, phase=BatchPhase.DECODE)
        batch.input_ids = mx.array([], dtype=mx.int32)
        batch.out_loc = out_loc
        batch.padded_reqs = reqs
        batch.attn_metadata = metadata
        return batch

    def _build_draft_calibration_batch(
        self,
        reqs: List[Req],
        drafts: mx.array,
        bonus_tokens: mx.array,
        verify_hidden: mx.array,
        slabs: mx.array,
        num_accepted_host: List[int],
    ) -> Tuple[Batch, mx.array, mx.array]:
        """Build the per-iter draft *calibration* prefill batch.

        For each req ``i`` with ``j_i = num_drafts_accepted``, builds a
        prefill window of ``j_i + 1`` positions corresponding to draft
        logical positions ``c..c+j_i`` (``c = req.cached_len``).  Per
        position:

        ===========  ===============================  ====================
        position     input token (shifted convention) target_hidden source
        ===========  ===============================  ====================
        ``c+k``      ``D_{k+1}`` (k < j_i) / bonus    ``verify_hidden[i, k]``
        ===========  ===============================  ====================

        Page IDs mirror the first ``j_i + 1`` columns of each req's
        shared slab, so the draft K/V written here overlaps the same
        page IDs as the (kept) target K/V from verify.

        Args:
            reqs: Batch reqs (their ``cached_len`` is still pre-iter).
            drafts: ``[B, K]`` int32 — the K drafts sampled this iter.
            bonus_tokens: ``[B]`` int32 — target's bonus per req.
            verify_hidden: ``[B, K+1, D]`` — real target hidden from
                verify.  Slice ``[i, :j_i+1]`` feeds the draft.
            slabs: ``[B, K+1]`` int32 — page IDs reserved for this iter.
            num_accepted_host: ``[B]`` Python ints (j_i values).

        Returns:
            ``(batch, input_ids, target_hidden)``:
                * ``batch``: ``PREFILL``-phase batch wired with the
                  draft's attention metadata + per-req ``out_loc``.
                  ``batch.input_ids`` is a placeholder — the draft
                  model takes the real ``input_ids`` and
                  ``target_hidden_states`` directly as call args.
                * ``input_ids`` ``[sum(j_i+1)]``: concatenated draft
                  input tokens in batch order.
                * ``target_hidden`` ``[sum(j_i+1), D]``: matching
                  target hidden states.
        """
        input_parts: List[mx.array] = []
        target_hidden_parts: List[mx.array] = []
        out_loc_parts: List[mx.array] = []
        extend_lens: List[int] = []
        kv_lens: List[int] = []

        for i, r in enumerate(reqs):
            j = num_accepted_host[i]
            # D_1..D_j (possibly empty when j=0) + bonus.
            input_parts.append(
                mx.concatenate([drafts[i, :j], bonus_tokens[i : i + 1]])
            )
            target_hidden_parts.append(verify_hidden[i, : j + 1])
            out_loc_parts.append(slabs[i, : j + 1])
            extend_lens.append(j + 1)
            kv_lens.append(r.cached_len + j + 1)

        input_ids = mx.concatenate(input_parts)
        target_hidden = mx.concatenate(target_hidden_parts, axis=0)
        out_loc = mx.concatenate(out_loc_parts)

        metadata = self.draft_attn_backend.build_prefill_metadata(
            reqs, extend_lens=extend_lens, kv_lens=kv_lens,
        )
        batch = Batch(reqs=reqs, phase=BatchPhase.PREFILL)
        batch.input_ids = mx.array([], dtype=mx.int32)
        batch.out_loc = out_loc
        batch.padded_reqs = reqs
        batch.attn_metadata = metadata
        return batch, input_ids, target_hidden
