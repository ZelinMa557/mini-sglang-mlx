import mlx.core as mx
import mlx.nn as nn

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
        self.rope_impl = nn.RoPE(rotary_dim, base=base)

    def __call__(self, hidden_state: mx.array, position_ids: mx.array) -> mx.array:
        B, H, D = hidden_state.shape
        hidden_state = hidden_state.reshape(B, H, 1, D)
        hidden_state = self.rope_impl(hidden_state, offset=position_ids)
        return hidden_state.squeeze(2)