import mlx.core as mx
import mlx.nn as nn

@mx.compile(shapeless=True)
def silu_mul(x: mx.array, y: mx.array):
    return nn.silu(x) * y