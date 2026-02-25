# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_serve.core import get_global_ctx
from mlx_serve.layers.activations import swiglu
from mlx_serve.layers.rotary_embedding import RotaryEmbedding
from .base import BaseModelArgs

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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.qo_attn_dim = self.n_heads * head_dim
        self.scale = head_dim**-0.5
        self.layer_id = layer_id

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = RotaryEmbedding(head_size=head_dim, rotary_dim=head_dim, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(L, self.n_heads, -1))
        keys = self.k_norm(keys.reshape(L, self.n_kv_heads, -1))
        values = values.reshape(L, self.n_kv_heads, -1)

        ctx = get_global_ctx()
        metadata = ctx.batch.attn_metadata
        queries = self.rope(queries, metadata.positions)
        keys = self.rope(queries, metadata.positions)
        output = ctx.attn_backend.forward(queries, keys, values, self.layer_id, ctx.batch)
        output = output.reshape(-1, self.qo_attn_dim)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_id)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x))
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_id=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
    ):
        print("inputs", inputs)
        h = self.embed_tokens(inputs)
        for layer in self.layers:
            h = layer(h)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
    ):
        out = self.model(get_global_ctx().batch.input_ids)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers