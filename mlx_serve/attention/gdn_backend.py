"""Backend for GatedDeltaNet linear attention layers.

Mirrors :class:`AttnBackend` but manages recurrent state (conv + temporal)
instead of paged KV cache. Both prefill and decode are dispatched through the
same fused slot-indexed recurrence kernel.

The backend only handles the recurrence — norm and output projection stay
in the model layer, mirroring how :class:`AttnBackend` returns raw attention
output and lets the :class:`Attention` layer apply ``o_proj``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_serve_kernel import gdn_state_inplace, gdn_state_verify  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from mlx_serve.core import Batch
    from mlx_serve.kvcache.mamba_pool import MambaStatePool

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
        """Run GDN recurrence for target-verify (MTP) phase.

        Each sequence processes ``num_draft`` tokens sequentially.
        The initial state is read from ``batch.mamba_slot_ids[b, 0]``;
        after token *j* the updated state is written to
        ``batch.mamba_slot_ids[b, j]`` so the caller can rollback to
        the last accepted token after verification.

        All sequences must have the same ``num_draft``.
        """
        temporal_buf = self.mamba_pool.temporal_state(linear_layer_idx)
        assert batch.mamba_slot_ids is not None

        return gdn_state_verify(
            q, k, v, g, beta,
            temporal_buf,
            batch.mamba_slot_ids,
        )
