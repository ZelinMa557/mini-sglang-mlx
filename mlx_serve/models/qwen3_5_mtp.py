"""MTP (Multi-Token Prediction) draft model for Qwen3.5 / Qwen3.6.

The draft model shares ``embed_tokens`` and ``lm_head`` with the target model.
It consumes the current token embedding plus the target model's hidden state
at the same position, projects the concatenation, runs a single full-attention
decoder layer, and produces logits for the next draft token.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple, Type

import mlx.core as mx
import mlx.nn as nn


def _mtp_args_from_target(target_args) -> "BaseModelArgs":
    """Derive MTP args from the target model's args.

    MTP always uses exactly one decoder layer with full attention.
    """
    return replace(
        target_args,
        num_hidden_layers=1,
        full_attention_interval=1,
    )


class Qwen3_5_MTPDraftModel(nn.Module):
    """Single-layer draft model for speculative multi-token prediction.

    Args:
        args: Model configuration (usually derived from the target model).
        decoder_layer_cls: The decoder layer class to instantiate
            (e.g. ``qwen3_5.DecoderLayer`` or ``qwen3_5_moe.DecoderLayer``).
    """

    def __init__(self, args, decoder_layer_cls: Type[nn.Module]):
        super().__init__()
        self.args = args
        dim = args.hidden_size

        self.pre_fc_norm_embedding = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * dim, dim, bias=False)

        # One full-attention decoder layer (full_attention_interval=1 guarantees
        # layer_idx=0 is an Attention layer, not GatedDeltaNet).
        # Pass attn_layer_idx=0 explicitly so the Attention sub-module's
        # ``layer_id`` matches the draft KV cache's single-layer indexing.
        self.layers = [decoder_layer_cls(args=args, layer_idx=0, attn_layer_idx=0)]

        self.norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)

        # Populated later by sharing target model weights.
        self.embed_tokens: nn.Embedding | None = None
        self.lm_head: nn.Linear | None = None

    def __call__(
        self,
        input_ids: mx.array,
        target_hidden_states: mx.array,
        return_hidden: bool = False,
    ):
        """Forward one draft step.

        Args:
            input_ids: Token IDs for the current position ``[L]``.
            target_hidden_states: Hidden states from the target model at the
                same position ``[L, hidden_size]``.
            return_hidden: When ``True``, also return the draft's post-norm
                hidden states ``[L, hidden_size]``.  These act as the
                ``target_hidden_states`` proxy for the *next* draft step
                in EAGLE-style multi-step speculative decoding.

        Returns:
            ``logits`` ``[L, vocab_size]`` or ``(hidden, logits)`` when
            ``return_hidden=True``.
        """
        assert self.embed_tokens is not None
        input_embeds = self.embed_tokens(input_ids)

        input_embeds = self.pre_fc_norm_embedding(input_embeds)
        hidden_states = self.pre_fc_norm_hidden(target_hidden_states)
        hidden_states = mx.concatenate([input_embeds, hidden_states], axis=-1)
        hidden_states = self.fc(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)

        if self.args.tie_word_embeddings:
            assert self.embed_tokens is not None
            logits = self.embed_tokens.as_linear(hidden_states)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden_states)
        if return_hidden:
            return hidden_states, logits
        return logits

    # ------------------------------------------------------------------ #
    # Weight loading helpers
    # ------------------------------------------------------------------ #

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap MTP checkpoint keys to this module's parameter names.

        The checkpoint uses a flat ``mtp.*`` prefix.  We strip it and handle
        MoE fused-expert sharding so that ``load_weights`` succeeds.
        """
        sanitized: Dict[str, mx.array] = {}

        # Detect MoE vs dense by inspecting raw keys.
        is_moe = any("mlp.experts" in k for k in weights.keys())

        for k, v in weights.items():
            # Keep only MTP branch weights
            if not k.startswith("mtp."):
                continue

            # Remove mtp. prefix
            k = k[4:]

            # Checkpoint stores attention weights under layers.0.self_attn.*
            # which already matches our DecoderLayer, so no remapping needed.

            # --- MoE fused experts ---
            if is_moe:
                if ".mlp.experts.gate_up_proj" in k:
                    w_gate, w_up = mx.split(v, 2, axis=-2)
                    base = k.replace(".mlp.experts.gate_up_proj", ".mlp.switch_mlp")
                    sanitized[f"{base}.gate_proj.weight"] = w_gate
                    sanitized[f"{base}.up_proj.weight"] = w_up
                    continue

                if ".mlp.experts.down_proj" in k:
                    k = k.replace(".mlp.experts.down_proj", ".mlp.switch_mlp.down_proj.weight")

            # --- Dense MLP ---
            else:
                if ".mlp.experts.gate_up_proj" in k:
                    w_gate, w_up = mx.split(v, 2, axis=-2)
                    base = k.replace(".mlp.experts.gate_up_proj", ".mlp")
                    sanitized[f"{base}.gate_proj.weight"] = w_gate
                    sanitized[f"{base}.up_proj.weight"] = w_up
                    continue

                if ".mlp.experts.down_proj" in k:
                    k = k.replace(".mlp.experts.down_proj", ".mlp.down_proj.weight")

            sanitized[k] = v

        # Same conv1d / norm sanitization as the target Qwen3.5 model
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1
            for k, v in sanitized.items()
        )

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )

        for k in list(sanitized.keys()):
            v = sanitized[k]
            if "conv1d.weight" in k and v.shape[-1] != 1:
                sanitized[k] = v.moveaxis(2, 1)
            if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    sanitized[k] = v + 1.0

        return sanitized


def load_qwen3_5_mtp_draft_model(
    mtp_path: str | Path,
    target_model: nn.Module,
) -> Tuple[Qwen3_5_MTPDraftModel, "BaseModelArgs"]:
    """Load an MTP draft model, sharing embed / lm_head with *target_model*.

    Args:
        mtp_path: Path to the MTP ``.safetensors`` file or a directory
            containing ``model*.safetensors``.
        target_model: The already-loaded target model.

    Returns:
        (draft_model, mtp_args)
    """
    import glob

    from mlx_serve.models.qwen3_5 import DecoderLayer as DenseDecoderLayer
    from mlx_serve.models.qwen3_5_moe import DecoderLayer as MoeDecoderLayer

    mtp_path = Path(mtp_path)

    # ------------------------------------------------------------------ #
    # 1. Detect target architecture (dense vs MoE)
    # ------------------------------------------------------------------ #
    first_layer = target_model.layers[0]
    if hasattr(first_layer.mlp, "switch_mlp"):
        decoder_layer_cls = MoeDecoderLayer
    else:
        decoder_layer_cls = DenseDecoderLayer

    # ------------------------------------------------------------------ #
    # 2. Build MTP args (1 layer, full attention)
    # ------------------------------------------------------------------ #
    target_args = target_model.args
    mtp_args = _mtp_args_from_target(target_args)

    # ------------------------------------------------------------------ #
    # 3. Load raw weights
    # ------------------------------------------------------------------ #
    if mtp_path.is_file():
        weight_files = [str(mtp_path)]
    else:
        weight_files = sorted(glob.glob(str(mtp_path / "model*.safetensors")))

    if not weight_files:
        raise FileNotFoundError(f"No MTP safetensors found at {mtp_path}")

    weights: Dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # ------------------------------------------------------------------ #
    # 4. Instantiate + sanitize + load
    # ------------------------------------------------------------------ #
    draft = Qwen3_5_MTPDraftModel(mtp_args, decoder_layer_cls)
    weights = draft.sanitize(weights)
    draft.load_weights(list(weights.items()), strict=True)
    mx.eval(draft.parameters())

    # ------------------------------------------------------------------ #
    # 5. Share embed_tokens and lm_head
    # ------------------------------------------------------------------ #
    draft.embed_tokens = target_model.model.embed_tokens
    if target_args.tie_word_embeddings:
        draft.lm_head = None
    else:
        draft.lm_head = target_model.lm_head

    return draft, mtp_args
