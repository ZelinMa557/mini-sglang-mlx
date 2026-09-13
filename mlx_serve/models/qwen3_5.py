"""Qwen3.5 dense hybrid model: interleaved full-attention + GatedDeltaNet layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from mlx_serve.core import Batch
    from mlx_serve.kvcache.mamba_pool import MambaStatePool

import mlx.core as mx
import mlx.nn as nn

from mlx_serve.core import get_global_ctx
from mlx_serve.layers.gated_delta import RMSNormGated, compute_gate

from mlx_serve.layers.rotary_embedding import RotaryEmbedding

from .base import BaseModelArgs
from .qwen3 import MLP


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, params):
        if "text_config" in params:
            text = params["text_config"]
            text["model_type"] = params.get("model_type", text.get("model_type", ""))
            params = text
        if "rope_parameters" in params:
            rp = params["rope_parameters"]
            params.setdefault("rope_theta", rp.get("rope_theta", 100000.0))
            params.setdefault("partial_rotary_factor", rp.get("partial_rotary_factor", 0.25))
        return super().from_dict(params)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.layer_id = layer_id

        # q_proj produces 2x: queries + gate
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rotary_dim = int(self.head_dim * args.partial_rotary_factor)
        self.rope = RotaryEmbedding(
            head_size=self.head_dim, rotary_dim=rotary_dim, base=args.rope_theta,
        )

    def __call__(self, x: mx.array) -> mx.array:
        L = x.shape[0]

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(L, self.n_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(L, -1)  # [L, n_heads * head_dim]

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys.reshape(L, self.n_kv_heads, -1))
        values = values.reshape(L, self.n_kv_heads, -1)

        ctx = get_global_ctx()
        metadata = ctx.batch.attn_metadata
        queries = self.rope(queries, metadata.positions)
        keys = self.rope(keys, metadata.positions)

        output = ctx.attn_backend.forward(queries, keys, values, self.layer_id, ctx.batch)
        output = output.reshape(L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class GatedDeltaNet(nn.Module):
    """GatedDeltaNet adapted for ragged tensor layout used by mlx-serve.

    Handles weight-bound projections, then delegates state-dependent
    conv + recurrence to :class:`GDNBackend`.
    """

    def __init__(self, args: ModelArgs, layer_idx: int, linear_layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.linear_layer_idx = linear_layer_idx

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False,
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    @property
    def conv_state_shape(self) -> tuple:
        return (self.conv_kernel_size - 1, self.conv_dim)

    @property
    def temporal_state_shape(self) -> tuple:
        return (self.num_v_heads, self.head_v_dim, self.head_k_dim)

    def _apply_conv_prefill(
        self, qkv: mx.array, batch: "Batch", mamba_pool: "MambaStatePool",
    ) -> mx.array:
        """Per-request depthwise conv1d using conv state from pool."""
        conv_buf = mamba_pool.conv_state(self.linear_layer_idx)
        state_len = self.conv_kernel_size - 1

        output_parts: List[mx.array] = []
        offset = 0

        for req in batch.reqs:
            seg_len = req.extend_len
            qkv_seg = qkv[offset : offset + seg_len]
            offset += seg_len

            slot = req.mamba_slot
            assert slot is not None
            conv_state = conv_buf[slot]  # [K-1, conv_dim]
            conv_input = mx.concatenate([conv_state, qkv_seg], axis=0)
            conv_buf[slot] = conv_input[-state_len:]

            conv_out = self.conv1d(conv_input[None, :, :])[0]
            output_parts.append(nn.silu(conv_out))

        return mx.concatenate(output_parts, axis=0)

    def _apply_conv_decode(
        self, qkv: mx.array, batch: "Batch", mamba_pool: "MambaStatePool",
    ) -> mx.array:
        """Batched depthwise conv1d for decode (1 token per request)."""
        assert batch.mamba_slot_ids is not None
        conv_buf = mamba_pool.conv_state(self.linear_layer_idx)
        state_len = self.conv_kernel_size - 1
        conv_state = conv_buf[batch.mamba_slot_ids]  # [B, K-1, conv_dim]
        conv_input = mx.concatenate([conv_state, qkv[:, None, :]], axis=1)
        conv_buf[batch.mamba_slot_ids] = conv_input[:, -state_len:, :]

        conv_out = self.conv1d(conv_input)  # [B, 1, conv_dim]
        return nn.silu(conv_out.squeeze(axis=1))

    def _apply_conv_verify(
        self, qkv: mx.array, batch: "Batch", mamba_pool: "MambaStatePool",
    ) -> mx.array:
        """Target-verify (replay-style): batched depthwise conv1d over the
        ``W = K + 1`` token window starting from the scratch slot's conv
        state (a copy of the main slot made before the verify forward).

        The per-step sliding windows are checkpointed into the small
        window buffer (row ``[scratch_slot, j]`` = window after ``j + 1``
        tokens) so ``GDNBackend.commit_verify`` can restore the window
        after the accepted prefix without storing full per-token states.
        """
        assert batch.mamba_scratch_slots is not None
        conv_buf = mamba_pool.conv_state(self.linear_layer_idx)
        window_buf = mamba_pool.conv_window(self.linear_layer_idx)
        state_len = self.conv_kernel_size - 1

        scratch_slots = batch.mamba_scratch_slots  # [batch_size]
        batch_size = scratch_slots.shape[0]
        num_draft = qkv.shape[0] // batch_size

        # 1. Gather base conv states: [batch_size, K-1, conv_dim]
        conv_state = conv_buf[scratch_slots]

        # 2. Reshape qkv to [batch_size, num_draft, conv_dim]
        seg = qkv.reshape(batch_size, num_draft, self.conv_dim)

        # 3. Batched conv1d: [batch_size, K-1+num_draft, conv_dim] -> [batch_size, num_draft, conv_dim]
        conv_input = mx.concatenate([conv_state, seg], axis=1)
        conv_out = self.conv1d(conv_input)
        conv_out = conv_out.reshape(batch_size * num_draft, self.conv_dim)

        # 4. Save per-step conv windows: row [b, j] is the window after
        #    ``j + 1`` tokens, i.e. ``conv_input[:, j+1 : j+1+state_len]``.
        #    One gather over the whole window instead of a per-step loop.
        offsets = mx.arange(num_draft)[:, None] + mx.arange(state_len)[None, :] + 1
        window_buf[scratch_slots] = mx.take(conv_input, offsets, axis=1)

        return nn.silu(conv_out)

    def __call__(self, x: mx.array) -> mx.array:
        ctx = get_global_ctx()
        batch = ctx.batch
        gdn_backend = ctx.gdn_backend
        mamba_pool = ctx.mamba_pool
        assert gdn_backend is not None and mamba_pool is not None

        L = x.shape[0]

        mixed_qkv = self.in_proj_qkv(x)                       # [L, conv_dim]
        z = self.in_proj_z(x).reshape(L, -1, self.head_v_dim)  # [L, Hv, Dv]
        b = self.in_proj_b(x)                                  # [L, Hv]
        a = self.in_proj_a(x)                                  # [L, Hv]
        if batch.is_decode:
            conv_out = self._apply_conv_decode(mixed_qkv, batch, mamba_pool)
        elif batch.is_target_verify:
            conv_out = self._apply_conv_verify(mixed_qkv, batch, mamba_pool)
        else:
            conv_out = self._apply_conv_prefill(mixed_qkv, batch, mamba_pool)
        q = conv_out[:, : self.key_dim].reshape(L, self.num_k_heads, self.head_k_dim)
        k = conv_out[:, self.key_dim : 2 * self.key_dim].reshape(L, self.num_k_heads, self.head_k_dim)
        v = conv_out[:, 2 * self.key_dim :].reshape(L, self.num_v_heads, self.head_v_dim)

        inv_scale = self.head_k_dim ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        beta = mx.sigmoid(b)
        g = compute_gate(self.A_log, a, self.dt_bias)

        if batch.is_target_verify:
            y = gdn_backend.forward_verify(
                q, k, v, g, beta,
                self.linear_layer_idx, batch,
            )
        else:
            y = gdn_backend.forward(
                q, k, v, g, beta,
                self.linear_layer_idx, batch,
            )
        y = self.norm(y, z)
        return self.out_proj(y.reshape(L, -1))


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
        self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x))
        else:
            r = self.self_attn(self.input_layernorm(x))
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5Model(nn.Module):
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
        """Run the inner decoder.

        Args:
            inputs: 1-D int32 token IDs of the active batch's flat
                input window (concat across reqs).
            capture_layer_ids: Optional layer indices (0-based, into
                ``self.layers``) whose hidden states should also be
                returned.  When set, returns
                ``(final_hidden, captured_list)``; ``captured_list[k]``
                is the hidden state captured at
                ``self.layers[capture_layer_ids[k]]``.

                **Last-layer special case** (intentional API quirk).
                For ``i = len(self.layers) - 1`` the captured tensor
                is the *post-norm* hidden — i.e. the same tensor that
                goes into ``lm_head``.  For every other layer it's
                the pre-final-norm residual stream output of that
                layer.  This way EAGLE-style draft engines, which
                want the LM-head input, get the right tensor simply
                by asking for the last layer; DFlash-style engines,
                which fan a few middle layers into ``fc``, get raw
                residual-stream outputs as expected.

                IDs must be unique; the returned list matches the
                caller's input order (no implicit sort).
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
                # Apply final norm in-place so both the captured
                # last-layer tensor and the function's return value
                # share the same post-norm hidden (used by lm_head).
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
        self.model = Qwen3_5Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        capture_layer_ids: Tuple[int, ...] | None = None,
    ):
        """Run the target model on the active batch's input_ids.

        Args:
            capture_layer_ids: Optional per-layer hidden capture (see
                :meth:`Qwen3_5Model.__call__` for details, including
                the last-layer post-norm special case).  When set,
                returns ``(captured_list, logits)``; otherwise just
                ``logits``.  EAGLE-style draft engines pass
                ``(num_layers - 1,)`` to grab the LM-head input.
                DFlash-style engines pass several middle-layer
                indices and concatenate the result along the
                feature dim.
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
