import mlx.core as mx
import mlx.nn as nn
from mlx_serve_kernel import fused_add_rmsnorm


class RMSNorm(nn.Module):

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def __call__(self, x: mx.array, y: mx.array = None):
        if y is None:
            return mx.fast.rms_norm(x, self["weight"], self.eps)
        return fused_add_rmsnorm(x, y, self["weight"], self.eps)
