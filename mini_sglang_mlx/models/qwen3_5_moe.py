from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from mini_sglang_mlx.core import Batch
    from mini_sglang_mlx.kvcache.mamba_pool import MambaStatePool

import mlx.core as mx
import mlx.nn as nn

from mini_sglang_mlx.core import get_global_ctx
from mini_sglang_mlx.layers.switch_linear import SwitchGLU

from .base import BaseModelArgs
from .qwen3 import MLP as Qwen3_5MoeMLP
from .qwen3_5 import (
    Attention,
    GatedDeltaNet,
    ModelArgs as BaseModelArgs,
    Qwen3_5Model,
)


@dataclass
class ModelArgs(BaseModelArgs):
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        self.shared_expert = Qwen3_5MoeMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds, scores)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y


class DecoderLayer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_idx: int,
        *,
        linear_layer_idx: int = -1,
        attn_layer_idx: int = -1,
    ):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0

        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args, layer_idx, linear_layer_idx)
        else:
            self.self_attn = Attention(args, attn_layer_idx)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3_5MoeSparseMoeBlock(args)

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x))
        else:
            r = self.self_attn(self.input_layernorm(x))
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        linear_counter = 0
        attn_counter = 0
        layers = []
        for i in range(args.num_hidden_layers):
            is_linear = (i + 1) % args.full_attention_interval != 0
            if is_linear:
                layers.append(DecoderLayer(
                    args=args, layer_idx=i, linear_layer_idx=linear_counter,
                ))
                linear_counter += 1
            else:
                layers.append(DecoderLayer(
                    args=args, layer_idx=i, attn_layer_idx=attn_counter,
                ))
                attn_counter += 1
        self.layers = layers
        self.num_attn_layers = attn_counter
        self.num_linear_layers = linear_counter

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        capture_layer_ids: Tuple[int, ...] | None = None,
    ):
        """Run the inner MoE decoder.

        See :meth:`qwen3_5.Qwen3_5Model.__call__` for ``capture_layer_ids``
        semantics, including the last-layer post-norm special case —
        same behaviour here.
        """
        h = self.embed_tokens(inputs)
        captured: List[mx.array] = []
        capture_set = (
            None if not capture_layer_ids else set(capture_layer_ids)
        )
        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i == last_idx:
                h = self.norm(h)
            if capture_set is not None and i in capture_set:
                captured.append(h)
        if capture_layer_ids is None:
            return h
        assert len(captured) == len(set(capture_layer_ids)), (
            f"capture_layer_ids contains duplicates: {capture_layer_ids}"
        )
        order_map = {
            lid: pos for pos, lid in enumerate(sorted(set(capture_layer_ids)))
        }
        captured_in_order = [
            captured[order_map[lid]] for lid in capture_layer_ids
        ]
        return h, captured_in_order


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5MoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        capture_layer_ids: Tuple[int, ...] | None = None,
    ):
        """Run the target model on the active batch's input_ids.

        See :meth:`mini_sglang_mlx.models.qwen3_5.Model.__call__` for the
        full contract — same kwargs / return shapes here.
        """
        if capture_layer_ids is not None:
            hidden, captured = self.model(
                get_global_ctx().batch.input_ids,
                capture_layer_ids=capture_layer_ids,
            )
        else:
            hidden = self.model(get_global_ctx().batch.input_ids)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        if capture_layer_ids is not None:
            return captured, logits
        return logits

    @property
    def is_hybrid(self) -> bool:
        return True

    def get_linear_layer_indices(self) -> List[int]:
        return [
            i for i, layer in enumerate(self.model.layers)
            if layer.is_linear
        ]

    def get_linear_state_shapes(self) -> tuple:
        """Return (conv_shapes, temporal_shapes) for MambaStatePool config."""
        conv_shapes: List[List[tuple]] = []
        temporal_shapes: List[tuple] = []
        for layer in self.model.layers:
            if layer.is_linear:
                gdn = layer.linear_attn
                conv_shapes.append([gdn.conv_state_shape])
                temporal_shapes.append(gdn.temporal_state_shape)
        return conv_shapes, temporal_shapes

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if k.startswith("language_model.model."):
                k = k.replace("language_model.model.", "model.")
            elif k.startswith("language_model."):
                k = k.replace("language_model.", "")
            elif k.startswith("vision_tower"):
                continue
            sanitized[k] = v
        weights = sanitized

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )

        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )

        for k in list(weights.keys()):
            v = weights[k]
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def layers(self):
        return self.model.layers
