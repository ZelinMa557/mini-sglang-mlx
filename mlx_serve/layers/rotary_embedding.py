import mlx.core as mx
import mlx.nn as nn

from mlx_serve_kernel import varlen_rope
from mlx_serve.engine.forward_batch import ForwardBatch


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        base: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.base = base

    def __call__(self, hidden_state: mx.array, forward_batch: ForwardBatch) -> mx.array:
        return varlen_rope(
            hidden_state,
            forward_batch.position_ids,
            self.rotary_dim,
            self.base
        )
