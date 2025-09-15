import mlx.core as mx
import mlx.nn as nn
from .base import BaseModelArgs
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from mlx_serve.engine.forward_batch import ForwardBatch
from mlx_serve.layers.rmsnorm import RMSNorm
from mlx_serve.layers.act import silu_mul


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


class Qwen3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gated = self.gate_proj(x)
        uped = self.up_proj(x)
        return self.down_proj(silu_mul(gated, uped))


class Qwen3Attention(nn.Module):
    pass


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Qwen3Attention(args)
        self.mlp = Qwen3MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        forward_batch: ForwardBatch,
    ) -> Tuple[mx.array, mx.array]:
        r = self.self_attn(self.input_layernorm(hidden_states, residual), forward_batch)
        h = hidden_states + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h, r


class Qwen3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3DecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.embed_tokens(inputs)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, forward_batch)

        return self.norm(hidden_states, residual)
