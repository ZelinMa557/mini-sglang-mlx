"""Backend for GatedDeltaNet linear attention layers.

Mirrors :class:`AttnBackend` but manages recurrent state (conv + temporal)
instead of paged KV cache.  Separates prefill (variable-length sequential
recurrence) from decode (batched single-step Metal kernel).

The backend only handles the recurrence — norm and output projection stay
in the model layer, mirroring how :class:`AttnBackend` returns raw attention
output and lets the :class:`Attention` layer apply ``o_proj``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_serve.core import Batch, Req
    from mlx_serve.kvcache.mamba_pool import MambaStatePool


# ── single-token recurrent step (used by both paths) ──────────────────────


def _gated_delta_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Single recurrent step, no batch dim.

    q, k: [Hk, Dk]   v: [Hv, Dv]   g, beta: [Hv]
    state: [Hv, Dv, Dk]
    """
    decay = g[:, None, None]
    hk_per_hv = state.shape[0] // q.shape[0]
    q_exp = mx.repeat(q, hk_per_hv, axis=0) if hk_per_hv > 1 else q
    k_exp = mx.repeat(k, hk_per_hv, axis=0) if hk_per_hv > 1 else k

    state = state * decay
    kv_mem = (state * k_exp[:, None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[:, None]
    state = state + k_exp[:, None, :] * delta[:, :, None]
    y = (state * q_exp[:, None, :]).sum(axis=-1)
    return y, state


# ── batched single-token step (ops fallback for decode) ───────────────────


@mx.compile
def _gated_delta_step_batched(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Batched single-step recurrence (ops path).

    q, k: [B, Hk, Dk]   v: [B, Hv, Dv]   g, beta: [B, Hv]
    state: [B, Hv, Dv, Dk]
    Returns y: [B, Hv, Dv], new_state: [B, Hv, Dv, Dk]
    """
    decay = g[:, :, None, None]
    Hv = state.shape[1]
    Hk = q.shape[1]
    hk_per_hv = Hv // Hk
    if hk_per_hv > 1:
        q = mx.repeat(q, hk_per_hv, axis=1)
        k = mx.repeat(k, hk_per_hv, axis=1)

    state = state * decay
    kv_mem = (state * k[:, :, None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[:, :, None]
    state = state + k[:, :, None, :] * delta[:, :, :, None]
    y = (state * q[:, :, None, :]).sum(axis=-1)
    return y, state


# ── Metal kernel for batched decode ───────────────────────────────────────

_decode_kernel_cache: dict = {}


def _get_decode_kernel(Hk: int, Hv: int, Dk: int, Dv: int):
    """Lazily build & cache the Metal kernel for batched GDN decode."""
    global _decode_kernel_cache
    key = (Hk, Hv, Dk, Dv)
    if key in _decode_kernel_cache:
        return _decode_kernel_cache[key]

    if not mx.metal.is_available():
        _decode_kernel_cache[key] = None
        return None

    source = f"""
        // Grid: (32, Dv, B * Hv)
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * Hv * Dv + hv_idx * Dv;
        y += b_idx * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        auto g_ = g + b_idx * Hv;
        auto beta_ = beta + b_idx * Hv;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }}

        // Single step — decay, delta update, output
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * g_[hv_idx];
          kv_mem += state[i] * k_[s_idx];
        }}
        kv_mem = simd_sum(kv_mem);

        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + k_[s_idx] * delta;
          out += state[i] * q_[s_idx];
        }}
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {{
          y[dv_idx] = static_cast<InT>(out);
        }}

        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """

    kernel = mx.fast.metal_kernel(
        name=f"gdn_decode_step_Hk{Hk}_Hv{Hv}_Dk{Dk}_Dv{Dv}",
        input_names=["q", "k", "v", "g", "beta", "state_in"],
        output_names=["y", "state_out"],
        source=source,
    )
    _decode_kernel_cache[key] = kernel
    return kernel


def _gated_delta_decode_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Batched single-step via Metal kernel.

    q, k: [B, Hk, Dk]   v: [B, Hv, Dv]   g, beta: [B, Hv]
    state: [B, Hv, Dv, Dk]
    """
    B, Hk, Dk = q.shape
    Hv, Dv = v.shape[1], v.shape[2]
    kernel = _get_decode_kernel(Hk, Hv, Dk, Dv)

    if kernel is None:
        return _gated_delta_step_batched(q, k, v, g, beta, state)

    return kernel(
        inputs=[q, k, v, g, beta, state],
        template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, q.dtype],
    )


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
        """Prepare decode-only metadata shared by all linear layers."""
        if batch.is_prefill or batch.mamba_slot_ids is not None:
            return

        slots = [req.mamba_slot for req in batch.reqs]
        assert all(slot is not None for slot in slots), "Missing mamba slot"
        batch.mamba_slot_ids = mx.array(slots, dtype=mx.int32)
        mx.eval(batch.mamba_slot_ids)

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
        self.prepare_batch(batch)
        if batch.is_prefill:
            return self._forward_prefill(
                q, k, v, g, beta, linear_layer_idx, batch,
            )
        else:
            return self._forward_decode(
                q, k, v, g, beta, linear_layer_idx, batch,
            )

    # ── prefill: sequential per-request recurrence ────────────────────

    def _forward_prefill(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        linear_layer_idx: int,
        batch: "Batch",
    ) -> mx.array:
        temporal_buf = self.mamba_pool.temporal_state(linear_layer_idx)

        output_parts: List[mx.array] = []
        offset = 0

        for req in batch.reqs:
            seg_len = req.extend_len
            slot = req.mamba_slot
            assert slot is not None

            q_seg = q[offset : offset + seg_len]
            k_seg = k[offset : offset + seg_len]
            v_seg = v[offset : offset + seg_len]
            g_seg = g[offset : offset + seg_len]
            beta_seg = beta[offset : offset + seg_len]

            state = temporal_buf[slot]
            ys = []
            for t in range(seg_len):
                y_t, state = _gated_delta_step(
                    q_seg[t], k_seg[t], v_seg[t], g_seg[t], beta_seg[t], state,
                )
                ys.append(y_t)

            temporal_buf[slot] = state
            output_parts.append(mx.stack(ys, axis=0))  # [S, Hv, Dv]
            offset += seg_len

        return mx.concatenate(output_parts, axis=0)

    # ── decode: batched single-step with Metal kernel ─────────────────

    def _forward_decode(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        linear_layer_idx: int,
        batch: "Batch",
    ) -> mx.array:
        assert batch.mamba_slot_ids is not None
        temporal_buf = self.mamba_pool.temporal_state(linear_layer_idx)
        state_batch = temporal_buf[batch.mamba_slot_ids]  # [B, Hv, Dv, Dk]

        y, new_state = _gated_delta_decode_kernel(q, k, v, g, beta, state_batch)

        temporal_buf[batch.mamba_slot_ids] = new_state
        return y
