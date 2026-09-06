"""DFlash (block-diffusion) speculative-decoding engine.

DFlash predicts ``K = block_size - 1`` drafts per iter from a
*single* draft forward — the draft input is ``[T, MASK, MASK, ...,
MASK]`` of length ``K + 1`` and the model is trained to fill in the
masked slots conditioned on the (concatenated, projected) target
hidden states at recently-committed positions.  Compared to the
multi-step EAGLE loop this is structurally simple:

============  =========  ============================================
phase         forwards   work
============  =========  ============================================
prefill       1 target   capture multi-layer target hidden over the
                         extend window, sample ``T_1``, **calibrate
                         the draft KV cache** with the per-layer
                         projection of the captured hidden.
decode iter   1 draft +  proposal forward (reads cached calibrated
              1 target   draft K/V + writes transient proposal K/V),
                         target verify (writes target K/V + captures
                         hidden), accept the longest matching prefix,
                         **calibrate** the draft cache for the
                         accepted positions, then free unused pages.
============  =========  ============================================

Invariant
---------
At every iter boundary the draft KV cache is *in sync* with the
target cache up through ``cached_len``: every draft K/V entry at
positions in ``[0, cached_len)`` is the projection of the
TARGET-derived hidden at the same position.  This is enforced by
calibrating at the **end** of each iter (and at the end of
prefill).  Consequence: ``Req`` carries no DFlash-specific spec
state besides ``pending_token``.

Mirrored page layout
--------------------
The draft KV cache shares page IDs with the target via the same
``page_table``.  Each draft layer has its own per-layer KV buffer
(multi-layer MHA), but the SAME page IDs are used at the SAME
positions for both target and draft — letting the scheduler's
:class:`CacheManager` (radix or naive) own all page lifecycle
decisions, prefix cache included.

Per-iter page lifecycle (per req ``i``, ``j = num_drafts_accepted``):

* **Pre-iter**: allocate ``slabs[i, 0..K]`` (K+1 pages), stamp
  ``page_table[r, c..c+K]``.
* **Proposal forward**: draft writes transient K/V at
  ``slabs[i, 0..K]``.
* **Target verify**: writes target K/V at ``slabs[i, 0..K]``.
* **End-of-iter calibration**: overwrites the draft K/V at
  ``slabs[i, 0..j]`` (the accepted prefix) with target-derived
  K/V; the unused tail ``slabs[i, j+1..K]`` is still stale draft
  but is about to be freed.
* **Free**: ``slabs[i, j+1..K]`` (rejected tail) — both target and
  draft K/V at those page IDs return to the pool together.

Limitations (intentional first cut):
    * Greedy verification only (temperature 0).
    * No chunked prefill.
    * **Block attention kernel pending.**  The draft layers call
      :meth:`AttnBackend.forward` with new
      ``is_cross_attention`` / ``sliding_window_size`` kwargs that
      ``mlx_serve_kernel.paged_prefill_attention`` does not yet
      honour.  See ``docs/dflash_kernel_requirements.md`` for the
      required contract — once the kernel update lands this engine
      will run as-is.
"""

from __future__ import annotations

from typing import List

import mlx.core as mx

from mlx_serve.attention import AttnBackend
from mlx_serve.core import Batch, BatchPhase, Context, use_ctx
from mlx_serve.kvcache.mha_pool import MHAKVCache
from mlx_serve.models.dflash import load_dflash_draft_model
from mlx_serve.utils import init_logger

from .config import EngineConfig
from .engine import _ModelMeta
from .spec_engine import SpecEngine, SpecForwardOutput
from .spec_sample import GreedyVerifyResult, greedy_verify

logger = init_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# Engine
# ════════════════════════════════════════════════════════════════════


class DflashEngine(SpecEngine):
    """Block-diffusion speculative-decoding engine.

    The draft side is a multi-layer DFlash model (mix of full and
    sliding-window attention layers).  Shared with target:
    ``embed_tokens`` and ``lm_head``.  Shared via the mirrored
    layout: ``page_table``, page IDs (and consequently the prefix
    cache).

    The block-attention kernel is not yet implemented (see module
    docstring); this engine is otherwise complete and will run end-
    to-end once ``paged_prefill_attention`` learns to honour
    ``is_cross_attention`` / ``sliding_window_size``.
    """

    def __init__(self, config: EngineConfig):
        assert config.spec_algo == "dflash", (
            f"DflashEngine requires spec_algo='dflash', got {config.spec_algo!r}"
        )
        assert config.draft_path is not None, (
            "DflashEngine requires EngineConfig.draft_path."
        )
        assert config.num_draft_tokens >= 1, (
            f"DflashEngine requires num_draft_tokens >= 1, got "
            f"{config.num_draft_tokens}; this is the block_size - 1 "
            f"the draft checkpoint was trained with."
        )

        # Stash block_size BEFORE the base engine's __init__ runs so
        # the mamba pool sizing and the conv window buffers have K
        # available (Engine.__init__ → _create_mamba_pool → our
        # _extra_mamba_checkpoints_per_req / _verify_width).
        # block_size = K + 1: slot 0 of the block is the already-known
        # pending token T, slots 1..K are the masked positions the
        # draft fills in.
        self._block_size: int = config.num_draft_tokens + 1

        super().__init__(config)

        self.K: int = config.num_draft_tokens

        # ---- Load draft model + bind shared weights ------------------
        logger.info(
            "Loading DFlash draft model from %s", config.draft_path,
        )
        self.draft_model = load_dflash_draft_model(
            config.draft_path, self.model,
        )
        self.draft_config = self.draft_model.args
        logger.info(
            "DFlash draft loaded: block_size=%d K=%d target_layer_ids=%s "
            "num_layers=%d",
            self._block_size, self.K, self.draft_config.target_layer_ids,
            self.draft_model.num_layers,
        )

        self.mask_token_id: int = int(self.draft_config.mask_token_id)
        self.target_layer_ids: tuple = tuple(
            self.draft_config.target_layer_ids
        )
        max_layer = len(self.model.layers) - 1
        for lid in self.target_layer_ids:
            assert 0 <= lid <= max_layer, (
                f"draft target_layer_id {lid} out of range "
                f"[0, {max_layer}] for the target model."
            )

        # ---- Draft KV cache (multi-layer MHA, mirrored pages) --------
        self.draft_kv_cache = MHAKVCache(
            num_kv_heads=self.draft_model.num_kv_heads,
            num_layers=self.draft_model.num_layers,
            head_dim=self.draft_model.head_dim,
            num_pages=self.num_pages + 1,  # +1 mirrors target dummy slot
            dtype=self.dtype,
        )

        # Wrap the draft's per-head config in a _ModelMeta look-alike
        # so AttnBackend's per-head sm_scale uses the right head_dim.
        draft_meta = _ModelMeta(
            head_dim=self.draft_model.head_dim,
            num_kv_heads=self.draft_model.num_kv_heads,
            num_layers=self.draft_model.num_layers,
            vocab_size=self.model_meta.vocab_size,
            max_position=self.model_meta.max_position,
        )
        self.draft_attn_backend = AttnBackend(
            config=draft_meta,  # type: ignore[arg-type]
            kvcache=self.draft_kv_cache,
            page_table=self.page_table,
        )
        self.draft_ctx = Context(
            page_size=1,
            attn_backend=self.draft_attn_backend,
            mamba_pool=None,
            gdn_backend=None,
        )

    def _extra_mamba_checkpoints_per_req(self, config: EngineConfig) -> int:
        # One scratch state slot per req: target verify replays the K+1
        # window into the scratch slot; the accepted prefix is replayed
        # back into the main slot after verification (no per-token
        # snapshots — rollback is just a replay with a ragged length).
        return 1

    def _verify_width(self, config: EngineConfig) -> int:
        # Verify window = block_size = K + 1 positions per req.
        return self._block_size

    # ════════════════════════════════════════════════════════════════
    # Internal: calibrate the draft cache from captured target hidden.
    # ════════════════════════════════════════════════════════════════

    def _calibrate_draft(
        self,
        target_hidden_concat: mx.array,
        out_loc: mx.array,
        positions: mx.array,
    ) -> None:
        """Apply :meth:`Model.calibrate` under the draft context.

        Args mirror :meth:`mlx_serve.models.dflash.Model.calibrate`:

        * ``target_hidden_concat``: ``[N_ctx, num_capture * hidden_size]``
          concat'd over capture layers, packed in req order.
        * ``out_loc``: ``[N_ctx]`` int32 page IDs (= page_table
          entries at the same positions, mirrored layout).
        * ``positions``: ``[N_ctx]`` int32 RoPE positions in target
          sequence coordinates.

        No-op when ``out_loc`` is empty.
        """
        if target_hidden_concat.shape[0] == 0:
            return
        with use_ctx(self.draft_ctx):
            self.draft_model.calibrate(
                target_hidden_concat, out_loc, positions,
            )

    # ════════════════════════════════════════════════════════════════
    # Prefill: target prefill (capture) → sample T_1 → calibrate draft
    # ════════════════════════════════════════════════════════════════

    def _run_prefill(self, batch: Batch) -> SpecForwardOutput:
        from mlx_serve.scheduler.prefill import ChunkedReq  # local: avoid cycle

        reqs = batch.reqs
        B = len(reqs)
        for req in reqs:
            if isinstance(req, ChunkedReq):
                raise NotImplementedError(
                    "Chunked prefill is not supported by DflashEngine; "
                    "the calibration write needs the FULL extend window's "
                    "target hidden in one piece."
                )

        # 1. Target prefill, capturing hidden at the configured
        # target_layer_ids.  ``captured`` is a list of
        # ``[sum(extend_i), hidden_size]`` arrays — one per capture
        # layer, in the order given by ``target_layer_ids``.
        # ``batch.out_loc`` holds the page IDs allocated for the
        # extend positions (in batch / req order).
        self._prepare_target_prefill_inplace(batch)
        with self.ctx.forward_batch(batch):
            captured, target_logits = self.model(
                capture_layer_ids=self.target_layer_ids,
            )

        # 2. Sample T_1 per req at the last extend position.
        last_indices = batch.attn_metadata.get_last_indices(B)
        T_batch = mx.argmax(
            target_logits[last_indices], axis=-1,
        ).astype(mx.int32)  # [B]

        # 3. Calibrate the draft KV cache.
        # ``target_hidden_concat`` aligns 1:1 with ``batch.out_loc``
        # (both packed in req order, length = sum(extend_i)).
        # Calibration positions are ``[r.cached_len .. r.device_len)``
        # per req — i.e., the NEW positions the target just processed
        # (commit happens AFTER this step, so cached_len here is
        # still the pre-commit prefix-hit length M).
        target_hidden_concat = mx.concatenate(captured, axis=-1)
        calib_pos_parts: List[mx.array] = [
            mx.arange(r.cached_len, r.device_len, dtype=mx.int32)
            for r in reqs
        ]
        calib_positions = mx.concatenate(calib_pos_parts)
        self._calibrate_draft(
            target_hidden_concat, batch.out_loc, calib_positions,
        )

        # 4. Commit T_1 + pending state for first decode iter.
        accepted: List[mx.array] = [T_batch[i : i + 1] for i in range(B)]
        mx.eval(T_batch)
        for i, req in enumerate(reqs):
            n_committed = req.extend_len
            req.append_host(accepted[i])
            req.complete_many(n_committed)
            req.pending_token = accepted[i]
        return SpecForwardOutput(accepted_tokens=accepted)

    # ════════════════════════════════════════════════════════════════
    # Decode iter: proposal → verify → calibrate → commit
    # ════════════════════════════════════════════════════════════════

    def _run_decode_iter(self, batch: Batch) -> SpecForwardOutput:
        K = self.K
        reqs = batch.reqs
        B = len(reqs)
        verify_len = K + 1

        # ---- Pre-allocate B*(K+1) shared pages -----------------------
        # Target verify writes K+1 target K/V at ``slabs[:, 0..K]``;
        # the draft also writes its (transient) proposal K/V at the
        # same page IDs; we overwrite ``slabs[:, 0..j]`` with
        # calibrated draft K/V at
        # the end of this iter; the tail ``slabs[:, j+1..K]`` is
        # then freed.
        flat_pages, slabs = self._allocate_iter_slabs(reqs)

        # ---- Build proposal batch ------------------------------------
        # Per req: input_ids = [T, MASK*K] (length K+1).  T is the
        # known pending token (target's last output); the K mask
        # positions are predicted by the draft.
        proposal_input_ids = mx.concatenate(
            [
                mx.concatenate(
                    [
                        r.pending_token,  # type: ignore[list-item]
                        mx.full((K,), self.mask_token_id, dtype=mx.int32),
                    ]
                )
                for r in reqs
            ]
        )

        proposal_metadata = self.draft_attn_backend.build_prefill_metadata(
            reqs,
            extend_lens=[verify_len] * B,
            kv_lens=[r.cached_len + verify_len for r in reqs],
        )
        proposal_batch = Batch(reqs=reqs, phase=BatchPhase.PREFILL)
        proposal_batch.input_ids = proposal_input_ids
        proposal_batch.out_loc = flat_pages
        proposal_batch.padded_reqs = reqs
        proposal_batch.attn_metadata = proposal_metadata

        # ---- Draft proposal forward (NO calibration interleaved) ----
        # The draft KV cache is ALREADY calibrated up through
        # ``cached_len - 1`` (end-of-prev-iter calibration / end-of-
        # prefill calibration).  Proposal queries at positions
        # [c..c+K] write their own transient K/V at slabs[:, 0..K]
        # (used for self-attention within the block) and read cached
        # draft K/V at [0..c-1] from prior iters.
        with (
            use_ctx(self.draft_ctx),
            self.draft_ctx.forward_batch(proposal_batch),
        ):
            draft_logits_flat = self.draft_model(proposal_input_ids)
        V = draft_logits_flat.shape[-1]
        draft_logits = draft_logits_flat.reshape(B, verify_len, V)
        # Drafts come from slots 1..K (slot 0 is the known pending T).
        drafts = mx.argmax(
            draft_logits[:, 1:, :], axis=-1,
        ).astype(mx.int32)  # [B, K]

        # ---- Target verify on [T, d_1, ..., d_K] --------------------
        pending_T = mx.stack(
            [r.pending_token for r in reqs]  # type: ignore[misc]
        ).squeeze(-1)
        verify_input_ids = mx.concatenate(
            [pending_T[:, None], drafts], axis=1,
        ).reshape(-1)  # [B*(K+1)]
        verify_batch, scratch_mamba_slots = (
            self._build_target_verify_batch(
                reqs, verify_input_ids, flat_pages,
            )
        )
        with self.ctx.forward_batch(verify_batch):
            captured, verify_logits_flat = self.model(
                capture_layer_ids=self.target_layer_ids,
            )
        verify_logits = verify_logits_flat.reshape(B, verify_len, V)
        # captured[layer] is [B*(K+1), hidden_size] — reshape per-req.
        verify_hiddens_per_layer = [
            h.reshape(B, verify_len, -1) for h in captured
        ]

        # ---- Greedy acceptance + single host sync -------------------
        result: GreedyVerifyResult = greedy_verify(verify_logits, drafts)
        mx.eval(result.num_drafts_accepted, result.bonus_tokens, drafts)
        num_accepted_host: List[int] = result.num_drafts_accepted.tolist()
        bonus_host: List[int] = result.bonus_tokens.tolist()
        drafts_host: List[List[int]] = drafts.tolist()
        self._log_acceptance(num_accepted_host)

        # ---- Calibrate draft for the accepted positions -------------
        # Per req i with ``j_i = num_accepted``: write per-layer
        # projections of the verify-hidden at positions
        # ``[c_i .. c_i + j_i]`` to ``slabs[i, 0..j_i]``.  This
        # overwrites the transient proposal K/V with target-derived
        # K/V so the next iter sees a calibrated cache.
        calib_hidden_parts: List[mx.array] = []
        calib_out_loc_parts: List[mx.array] = []
        calib_pos_parts: List[mx.array] = []
        for i, r in enumerate(reqs):
            j = num_accepted_host[i]
            # ``[j+1, num_capture * hidden_size]`` for this req.
            per_layer = [
                h[i, : j + 1, :] for h in verify_hiddens_per_layer
            ]
            calib_hidden_parts.append(mx.concatenate(per_layer, axis=-1))
            calib_out_loc_parts.append(slabs[i, : j + 1])
            calib_pos_parts.append(
                mx.arange(r.cached_len, r.cached_len + j + 1, dtype=mx.int32)
            )
        if calib_hidden_parts:
            calib_target_hidden = mx.concatenate(calib_hidden_parts, axis=0)
            calib_out_loc = mx.concatenate(calib_out_loc_parts).astype(
                mx.int32,
            )
            calib_positions = mx.concatenate(calib_pos_parts)
            self._calibrate_draft(
                calib_target_hidden, calib_out_loc, calib_positions,
            )

        # ---- Release per-iter resources -----------------------------
        # AFTER calibration consumes verify_hidden + slabs; mamba
        # state and slab pages must stay live until here.
        self._release_iter_resources(
            reqs, num_accepted_host, slabs, verify_batch, scratch_mamba_slots,
        )

        # ---- Per-req state update + accepted_tokens output ----------
        accepted: List[mx.array] = []
        for i, r in enumerate(reqs):
            j = num_accepted_host[i]
            tail_list = drafts_host[i][:j] + [bonus_host[i]]
            tail = mx.array(tail_list, dtype=mx.int32)
            r.input_ids = mx.concatenate([r.input_ids, tail])
            r.complete_many(j + 1)
            r.pending_token = tail[-1:]
            accepted.append(tail)
        return SpecForwardOutput(accepted_tokens=accepted)
