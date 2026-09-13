"""Shared primitives for Gated Delta Net (used by Qwen3.5 linear attention layers).

Model-layer code lives in :mod:`mini_sglang_mlx.models.qwen3_5`.
Backend (prefill/decode dispatch, Metal kernel) lives in
:mod:`mini_sglang_mlx.attention.gdn_backend`.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_gate(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    """g = exp(-exp(A_log) * softplus(a + dt_bias)), cast back to input dtype."""
    return mx.exp(
        -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    ).astype(a.dtype)


class RMSNormGated(nn.Module):
    """RMSNorm with SiLU gating: out = rms_norm(x) * silu(z)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array, z: mx.array | None = None) -> mx.array:
        x = mx.fast.rms_norm(x, self.weight, self.eps)
        if z is not None:
            from mini_sglang_mlx.layers.activations import swiglu
            x = swiglu(z, x)
        return x
