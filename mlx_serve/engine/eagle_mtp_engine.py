"""EAGLE-style MTP (multi-token prediction) speculative-decoding engine.

This engine drives the Qwen3.5 MTP draft model alongside the target.
Each decode iter does:

    * ``K`` regular draft forwards (``K = num_mtp_step``) to produce
      drafts ``D_2..D_K`` and to commit the KV of every draft input
      ``D_1..D_K`` into the draft cache.  ``D_1`` itself comes from
      the previous iter / prefill.
    * A single target verify pass over ``[T, D_1, ..., D_K]`` (K+1
      tokens) that greedy-accepts the longest matching prefix.
    * One *bonus* draft forward on the freshly sampled bonus token to
      pre-stage the next iter's ``pending_draft_token`` /
      ``pending_draft_hidden`` (mirrors what prefill leaves behind).

Total draft forwards per decode iter: ``K + 1``.

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import mlx.core as mx

from mlx_serve.attention import AttnBackend
from mlx_serve.core import Batch, BatchPhase, Context, Req, use_ctx
from mlx_serve.kvcache.mha_pool import MHAKVCache
from mlx_serve.utils import init_logger

from .config import EngineConfig
from .engine import Engine
from .spec_sample import (
    GreedyVerifyResult,
    gather_last_accepted_hidden,
    greedy_verify,
)

if TYPE_CHECKING:
    from mlx_serve.scheduler.cache import CacheManager

logger = init_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# Output container
# ════════════════════════════════════════════════════════════════════


@dataclass
class SpecForwardOutput:
    """Per-iter MTP output handed back to the scheduler.

    ``accepted_tokens`` is a list of length ``B``; entry ``i`` is the
    1-D int32 array of NEW tokens committed for req ``i`` this iter.

    * Prefill: ``[T_1]`` (length 1).
    * Decode:  ``[D_1, ..., D_j, bonus]`` (length ``j+1``, with
      ``j = num_drafts_accepted``).  These are the tokens NOT already
      present in ``req.input_ids`` before the iter — they are appended
      to ``req.input_ids`` by the engine before returning.
    """

    accepted_tokens: List[mx.array]


# ════════════════════════════════════════════════════════════════════
# Engine
# ════════════════════════════════════════════════════════════════════


class EagleMTPEngine(Engine):
    """EAGLE-style MTP engine over the base :class:`Engine` skeleton.

    The base class loads the target + (optional) draft model and the
    target's MHA / Mamba pools.  This subclass additionally owns:

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
        super().__init__(config)
        assert self.draft_model is not None, (
            "EagleMTPEngine requires a draft model; "
            "set EngineConfig.mtp_model_path."
        )
        assert config.num_mtp_step >= 1, (
            f"num_mtp_step must be >= 1, got {config.num_mtp_step}"
        )
        self.K = config.num_mtp_step

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

        # Injected by the scheduler so the engine can alloc/free target
        # KV pages without bouncing through the scheduler on every
        # sub-forward.
        self._cache_manager: "CacheManager | None" = None

    def set_cache_manager(self, cache_manager: "CacheManager") -> None:
        self._cache_manager = cache_manager

    @property
    def cache_manager(self) -> "CacheManager":
        assert self._cache_manager is not None, (
            "EagleMTPEngine.cache_manager not set. "
            "Call set_cache_manager(...) after constructing the engine."
        )
        return self._cache_manager

    # ════════════════════════════════════════════════════════════════
    # Public entry point — called once per scheduled batch
    # ════════════════════════════════════════════════════════════════

    def run_iter(self, batch: Batch) -> SpecForwardOutput:
        """Drive one MTP iter (prefill OR decode)."""
        if batch.is_prefill:
            return self._run_prefill(batch)
        if batch.is_decode:
            return self._run_decode_iter(batch)
        raise ValueError(f"Unsupported phase for EagleMTPEngine: {batch.phase}")

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
            target_hidden, target_logits = self.model(return_hidden=True)

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
    # Decode iter: K regular draft steps → verify → bonus draft step
    # ════════════════════════════════════════════════════════════════

    def _run_decode_iter(self, batch: Batch) -> SpecForwardOutput:
        K = self.K
        reqs = batch.reqs
        B = len(reqs)
        verify_len = K + 1

        # ---- Pre-allocate B*(K+1) shared pages -----------------------
        # One slab per req, K+1 pages wide, covering positions
        # cached_len..cached_len+K.  The SAME page IDs are used by both
        # target (verify) and draft (K regular + 1 bonus step) — page i
        # of slab[r] simultaneously addresses target_kv_cache and
        # draft_kv_cache.  Page_table is stamped once here.
        flat_pages = self.cache_manager.allocate(B * verify_len)
        slabs = flat_pages.reshape(B, verify_len)
        for i, r in enumerate(reqs):
            c = r.cached_len
            self.page_table[r.table_idx, c : c + verify_len] = slabs[i]

        # ---- Gather pending state from prev iter ---------------------
        # ``stack(...).squeeze(-1)`` collapses a list of [1]-shape
        # tensors into a [B] tensor; pending_draft_hidden is [D] each.
        pending_T = mx.stack([r.pending_token for r in reqs]).squeeze(-1)
        cur_token = mx.stack([r.pending_draft_token for r in reqs]).squeeze(-1)
        cur_hidden = mx.stack([r.pending_draft_hidden for r in reqs])

        # ---- K regular draft steps -----------------------------------
        # D_1 = pending_draft_token came from prev iter (or prefill).
        # We run K draft forwards rather than K-1: the first K-1 produce
        # D_2..D_K, and the K-th step processes D_K as input to commit
        # its KV at draft slot ``cached_len + K - 1`` — required so the
        # bonus step's attention reads a valid slot when j=K (all drafts
        # accepted).  The K-th output is discarded.
        #
        # Step k writes draft K/V at page ID ``slabs[:, k]`` (= same ID
        # target_verify will write its K/V at for position cached_len+k).
        drafts_per_step: List[mx.array] = [cur_token]
        for step in range(K):
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
            next_token = mx.argmax(d_logits, axis=-1).astype(mx.int32)
            if step < K - 1:
                drafts_per_step.append(next_token)
            cur_token = next_token
        drafts = mx.stack(drafts_per_step, axis=1)  # [B, K]

        # ---- Target verify on [T, D_1, ..., D_K] --------------------
        verify_input_ids = mx.concatenate(
            [pending_T[:, None], drafts], axis=1
        ).reshape(-1)  # [B*(K+1)]
        verify_batch, slot_ids_rows, new_mamba_slots = (
            self._build_target_verify_batch(
                reqs, verify_input_ids, K, flat_pages,
            )
        )
        with self.ctx.forward_batch(verify_batch):
            verify_hidden_flat, verify_logits_flat = self.model(return_hidden=True)
        D = verify_hidden_flat.shape[-1]
        V = verify_logits_flat.shape[-1]
        verify_hidden = verify_hidden_flat.reshape(B, verify_len, D)
        verify_logits = verify_logits_flat.reshape(B, verify_len, V)

        # ---- Greedy acceptance --------------------------------------
        result: GreedyVerifyResult = greedy_verify(verify_logits, drafts)
        last_accepted_hidden = gather_last_accepted_hidden(
            verify_hidden, result.num_drafts_accepted
        )

        # Single host sync: counts + bonus + drafts in one eval so the
        # subsequent per-req bookkeeping doesn't trigger extra flushes.
        mx.eval(
            result.num_drafts_accepted,
            result.bonus_tokens,
            drafts,
            last_accepted_hidden,
        )
        num_accepted_host: List[int] = result.num_drafts_accepted.tolist()
        bonus_host: List[int] = result.bonus_tokens.tolist()
        drafts_host: List[List[int]] = drafts.tolist()

        # ---- Roll back mamba state ----------------------------------
        # slot_ids[i, 0] is the req's main slot; it was overwritten by
        # the verify kernel with "state after T". If j_i >= 1 drafts
        # were accepted, the correct post-iter state lives in
        # slot_ids[i, j_i] and we copy it back to the main slot.
        if self.is_hybrid:
            src_slots: List[int] = []
            dst_slots: List[int] = []
            for i, r in enumerate(reqs):
                j = num_accepted_host[i]
                if j >= 1:
                    src_slots.append(slot_ids_rows[i][j])
                    dst_slots.append(r.mamba_slot)  # type: ignore[arg-type]
            if src_slots:
                assert self.mamba_pool is not None
                self.mamba_pool.copy_batched(
                    mx.array(src_slots, dtype=mx.int32),
                    mx.array(dst_slots, dtype=mx.int32),
                )
            assert self.mamba_pool is not None
            self.mamba_pool.free_many(new_mamba_slots)

        # ---- Free unused (shared) pages -----------------------------
        # For req i: keep slabs[i, 0..j_i] (j_i+1 pages = accepted
        # prefix), free slabs[i, j_i+1..K] (K-j_i pages).  Both
        # target's and draft's K/V at those page IDs return to the pool
        # together — no separate "free draft pages" pass.
        free_chunks: List[mx.array] = []
        for i, j in enumerate(num_accepted_host):
            if K - j > 0:
                free_chunks.append(slabs[i, j + 1 :])
        if free_chunks:
            # ``CacheManager._free`` accepts batched indices.
            self.cache_manager._free(mx.concatenate(free_chunks))

        # ---- Bonus draft step ---------------------------------------
        # One extra draft step on the verified bonus token to pre-stage
        # D_1 + draft hidden for next iter (mirrors what prefill leaves
        # behind).  Writes draft K/V at slabs[i, j_i] (= page_table
        # entry for position cached_len+j_i), overwriting whatever the
        # regular step j_i wrote there (= D_{j_i+1}'s draft K/V).  The
        # target K/V at that same page ID stays intact (different
        # buffer), so the slot ends up with target K/V for D_{j_i} and
        # draft K/V for the bonus token — both are what the next iter
        # needs to read.
        j_arr = mx.array(num_accepted_host, dtype=mx.int32)
        bonus_out_loc = mx.take_along_axis(
            slabs, j_arr[:, None], axis=1,
        ).squeeze(axis=1)
        bonus_positions = [r.cached_len + num_accepted_host[i] for i, r in enumerate(reqs)]
        bonus_batch = self._build_draft_decode_batch(
            reqs, bonus_positions, bonus_out_loc,
        )
        with (
            use_ctx(self.draft_ctx),
            self.draft_ctx.forward_batch(bonus_batch),
        ):
            new_dh, new_dlogits = self.draft_model(
                result.bonus_tokens, last_accepted_hidden, return_hidden=True,
            )
        new_D1 = mx.argmax(new_dlogits, axis=-1).astype(mx.int32)
        mx.eval(new_D1, new_dh)

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
            r.pending_draft_hidden = new_dh[i]
            accepted.append(tail)
        return SpecForwardOutput(accepted_tokens=accepted)

    # ════════════════════════════════════════════════════════════════
    # Batch / metadata builders
    # ════════════════════════════════════════════════════════════════

    def _prepare_target_prefill_inplace(self, batch: Batch) -> None:
        """In-place setup of a target prefill batch.

        Mirrors what :meth:`Scheduler._prepare_batch` would do for the
        non-MTP engine.  Allocates target pages, writes page_table,
        concatenates input_ids and builds attn / mamba metadata.
        """
        reqs = batch.reqs
        needed = sum(r.extend_len for r in reqs)
        out_loc = self.cache_manager.allocate(needed)
        batch.out_loc = out_loc
        batch.padded_reqs = reqs

        offset = 0
        chunks: List[mx.array] = []
        for req in reqs:
            n = req.extend_len
            if n > 0:
                self.page_table[
                    req.table_idx, req.cached_len : req.device_len
                ] = out_loc[offset : offset + n]
                chunks.append(req.input_ids[req.cached_len : req.device_len])
                offset += n
        batch.input_ids = (
            mx.concatenate(chunks) if chunks else mx.array([], dtype=mx.int32)
        )
        self.attn_backend.prepare_metadata(batch)
        if self.gdn_backend is not None:
            self.gdn_backend.prepare_batch(batch)

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

    def _build_target_verify_batch(
        self,
        reqs: List[Req],
        verify_input_ids: mx.array,
        K: int,
        out_loc: mx.array,
    ) -> Tuple[Batch, List[List[int]], List[int]]:
        """Target verify batch with ``extend_len = K+1`` per req.

        ``out_loc`` is the pre-allocated, flattened ``[B*(K+1)]`` page
        IDs; the caller has already stamped them into the per-req
        page_table slices.

        Returns
            (batch, slot_ids_rows, new_mamba_slots)

        ``slot_ids_rows`` is the host-side list-of-lists mirror of
        ``batch.mamba_slot_ids`` (used to compute per-req rollback
        sources without an extra GPU→CPU sync).
        ``new_mamba_slots`` is the flat list of ``B*K`` checkpoint
        slots allocated this iter (freed in the caller after rollback).
        """
        B = len(reqs)
        verify_len = K + 1

        # ``extend_len`` per req is K+1, NOT req.extend_len — the target
        # evaluates ``[T, D_1, ..., D_K]`` over K+1 fresh positions.
        extend_lens = [verify_len] * B
        kv_lens = [r.cached_len + verify_len for r in reqs]
        metadata = self.attn_backend.build_prefill_metadata(
            reqs, extend_lens=extend_lens, kv_lens=kv_lens,
        )

        batch = Batch(reqs=reqs, phase=BatchPhase.TARGET_VERIFY)
        batch.input_ids = verify_input_ids
        batch.out_loc = out_loc
        batch.padded_reqs = reqs
        batch.attn_metadata = metadata

        slot_ids_rows: List[List[int]] = []
        new_mamba_slots: List[int] = []
        if self.is_hybrid:
            assert self.mamba_pool is not None
            allocated = self.mamba_pool.alloc_many(B * K)
            if allocated is None:
                raise RuntimeError(
                    f"Out of mamba checkpoint slots: needed {B * K}, "
                    f"have {self.mamba_pool.available_size}"
                )
            new_mamba_slots = allocated
            slot_ids_rows = [
                [r.mamba_slot] + new_mamba_slots[i * K : (i + 1) * K]  # type: ignore[list-item]
                for i, r in enumerate(reqs)
            ]
            batch.mamba_slot_ids = mx.array(slot_ids_rows, dtype=mx.int32)
            # No mamba_prefill_indptr needed for the verify path.

        return batch, slot_ids_rows, new_mamba_slots
