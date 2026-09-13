"""Backend for GatedDeltaNet linear attention layers.

Mirrors :class:`AttnBackend` but manages recurrent state (conv + temporal)
instead of paged KV cache. Both prefill and decode are dispatched through the
same fused slot-indexed recurrence kernel.

Target verify (speculative decoding) uses replay-style state management: the
whole ``K + 1`` window is replayed into one scratch slot per req (copying the
main slot first), then — once the accept counts are known — each req's
accepted prefix is replayed *again* into its main slot with a ragged length.
Rollback is therefore just a replay length; no per-token state snapshots are
ever stored, so the mamba pool only needs one scratch slot per req instead of
``K + 1`` full slots.

The backend only handles the recurrence — norm and output projection stay
in the model layer, mirroring how :class:`AttnBackend` returns raw attention
output and lets the :class:`Attention` layer apply ``o_proj``.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np

import mlx.core as mx
from mini_sglang_mlx_kernel import gdn_state_inplace

if TYPE_CHECKING:
    from mini_sglang_mlx.core import Batch
    from mini_sglang_mlx.kvcache.mamba_pool import MambaStatePool

# ── GDNBackend ────────────────────────────────────────────────────────────


class GDNBackend:
    """Manages GatedDeltaNet recurrence for all linear-attention layers.

    Analogous to :class:`AttnBackend` but for the GDN recurrence.
    The backend holds a reference to the :class:`MambaStatePool` and
    provides :meth:`forward` which the :class:`GatedDeltaNet` model layer
    calls with pre-projected q/k/v/g/beta tensors.  State read/write is
    handled internally.
    """

    def __init__(self, mamba_pool: MambaStatePool) -> None:
        self.mamba_pool = mamba_pool

    # ── public API called by GatedDeltaNet layer ──────────────────────

    def prepare_batch(self, batch: "Batch") -> None:
        """Prepare Mamba slot metadata shared by all linear layers."""
        if batch.mamba_slot_ids is None:
            slots = [req.mamba_slot for req in batch.reqs]
            assert all(slot is not None for slot in slots), "Missing mamba slot"
            batch.mamba_slot_ids = mx.array(slots, dtype=mx.int32)
            mx.eval(batch.mamba_slot_ids)

        # Reuse the existing field for both phases:
        # - prefill: ragged cumulative token offsets
        # - decode:  [0, 1, 2, ..., batch]
        if batch.mamba_prefill_indptr is None:
            indptr = [0]
            for req in batch.reqs:
                indptr.append(indptr[-1] + req.extend_len)
            batch.mamba_prefill_indptr = mx.array(indptr, dtype=mx.int32)
            mx.eval(batch.mamba_prefill_indptr)

    def forward(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        linear_layer_idx: int,
        batch: "Batch",
    ) -> mx.array:
        """Run GDN recurrence for one layer across the batch.

        Returns raw recurrence output ``y`` of shape ``[L, Hv, Dv]``.
        The caller is responsible for norm and output projection.

        Prefill: q/k/v/g/beta are ragged [total_tokens, ...].
        Decode:  q/k/v/g/beta are ragged [B, ...] (1 token each).
        """
        temporal_buf = self.mamba_pool.temporal_state(linear_layer_idx)
        assert batch.mamba_slot_ids is not None
        assert batch.mamba_prefill_indptr is not None

        return gdn_state_inplace(
            q, k, v, g, beta,
            temporal_buf,
            batch.mamba_slot_ids,
            batch.mamba_prefill_indptr,
            single_token_mode=batch.is_decode,
        )

    def forward_verify(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        linear_layer_idx: int,
        batch: "Batch",
    ) -> mx.array:
        """Replay the verify window into the per-req scratch slot.

        All sequences process the same ``W = K + 1`` token window
        (uniform ``batch.mamba_verify_indptr``), starting from the
        scratch slot — a copy of the main slot made before the
        verify forward.  The main slots are left untouched until
        :meth:`commit_verify` replays the accepted prefix back.

        The per-layer (q, k, v, g, beta) tensors are captured on the
        batch so :meth:`commit_verify` can gather each req's accepted
        prefix without recomputing the model projections.
        """
        temporal_buf = self.mamba_pool.temporal_state(linear_layer_idx)
        assert batch.mamba_scratch_slots is not None
        assert batch.mamba_verify_indptr is not None

        if batch.gdn_verify_captured is None:
            batch.gdn_verify_captured = {}
        batch.gdn_verify_captured[linear_layer_idx] = (q, k, v, g, beta)

        return gdn_state_inplace(
            q, k, v, g, beta,
            temporal_buf,
            batch.mamba_scratch_slots,
            batch.mamba_verify_indptr,
            single_token_mode=False,
        )

    def commit_verify(
        self,
        batch: "Batch",
        num_accepted_host: List[int],
    ) -> None:
        """Commit the accepted prefix of the verify window.

        For each req ``i`` with ``j_i`` accepted drafts, replay the
        first ``j_i + 1`` verify tokens from the scratch state back
        into the req's main slot, and restore the conv sliding window
        to the position after those tokens.  Reqs replay different
        lengths, so the inputs are gathered once per layer from the
        tensors captured during :meth:`forward_verify`.

        Called by the spec engine after verification (and any draft
        calibration that reads the verify outputs) and before the
        scratch slots are released.
        """
        captured = batch.gdn_verify_captured
        if not captured:
            return
        assert batch.mamba_slot_ids is not None
        assert batch.mamba_scratch_slots is not None
        assert batch.mamba_verify_indptr is not None

        # W = verify window width (K + 1), uniform across reqs.
        W = int(batch.mamba_verify_indptr[1].item())
        B = len(num_accepted_host)

        # Ragged replay layout: req i replays tokens
        # [i*W, i*W + j_i + 1) of the uniform [B*W] verify tensors.
        j_np = np.asarray(num_accepted_host, dtype=np.int32)
        lengths = j_np + 1
        parts = [
            np.arange(i * W, i * W + l, dtype=np.int32)
            for i, l in enumerate(lengths)
        ]
        # Single index array shared by every layer / tensor: the flat
        # token positions of the accepted prefix, in ragged order.
        idx = mx.array(np.concatenate(parts)) if parts else mx.zeros((0,), mx.int32)
        indptr = mx.array(
            np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
        )

        # Conv window rows: window after j_i + 1 tokens lives at row
        # j_i of the scratch slot's window buffer.
        j_idx = mx.array(j_np)

        main_slots = batch.mamba_slot_ids
        scratch_slots = batch.mamba_scratch_slots

        # The replay kernel mutates the state buffer as a side effect
        # and its y output is unused, so force evaluation of every
        # layer's replay (single sync per iter) before the scratch
        # slots are released.
        replay_outputs: List[mx.array] = []

        for linear_layer_idx, (q, k, v, g, beta) in captured.items():
            temporal_buf = self.mamba_pool.temporal_state(linear_layer_idx)
            y = gdn_state_inplace(
                mx.take(q, idx, axis=0),
                mx.take(k, idx, axis=0),
                mx.take(v, idx, axis=0),
                mx.take(g, idx, axis=0),
                mx.take(beta, idx, axis=0),
                temporal_buf,
                main_slots,
                indptr,
                single_token_mode=False,
            )
            replay_outputs.append(y)

            # Restore the conv window (group 0 — the model's GDN
            # layer uses a single conv group, matching prefill /
            # decode).  The window buffer rows were written by
            # ``_apply_conv_verify`` during the verify forward.
            conv_buf = self.mamba_pool.conv_state(linear_layer_idx, group=0)
            window_buf = self.mamba_pool.conv_window(linear_layer_idx, group=0)
            conv_buf[main_slots] = window_buf[scratch_slots, j_idx]

        if replay_outputs:
            mx.eval(*replay_outputs)

        # Drop the captured tensors so the verify buffers can be freed.
        batch.gdn_verify_captured = None
