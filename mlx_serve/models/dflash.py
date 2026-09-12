"""DFlash (block-diffusion) draft model.

DFlash predicts ``block_size - 1`` tokens per draft forward via
masked block diffusion: the draft input is ``[T, MASK, ..., MASK]``
of length ``block_size``, and the model is trained to recover the
masked slots conditioned on the (concatenated, projected) target
hidden states at recently-committed positions.

This module exposes the standard mlx-serve ``Model`` / ``ModelArgs``
pair so the generic :func:`mlx_serve.models.load_model` discovers
it via the usual ``model_type``-based dispatch.  The DFlash engine
adds the missing piece — binding ``embed_tokens`` / ``lm_head`` from
the target — via :func:`load_dflash_draft_model`.

* Embed-tokens + lm_head are SHARED with the target model.
* ``fc`` projects ``concat(target_hidden_at_capture_layers)`` (dim
  ``num_target_layers * hidden_size``) down to the draft's
  ``hidden_size``.
* Multi-layer attention with two kernel variants per layer:
    * ``full_attention``: bidirectional within the proposal block,
      full visibility over cached context.
    * ``sliding_attention``: causal with a sliding window.
  Both go through :meth:`AttnBackend.forward` (the standard prefill
  path) by setting ``is_cross_attention`` / ``sliding_window_size``;
  the underlying ``mlx_serve_kernel.paged_prefill_attention`` kernel
  must support those two flags — see
  ``docs/dflash_kernel_requirements.md`` for the contract.

The draft's KV cache mirrors the target's page layout (same
``num_pages``); each layer stores its own per-layer projection of
the target hidden, so the draft model itself is multi-layer
MHA-style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_serve.core import get_global_ctx
from mlx_serve.layers.rotary_embedding import RotaryEmbedding

from .base import BaseModelArgs
from .qwen3 import MLP


# ════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════


@dataclass
class ModelArgs(BaseModelArgs):
    """Draft-side config — loaded from the DFlash safetensors directory.

    Field names mirror the upstream reference for parity with the
    on-disk ``config.json`` schema.
    """

    model_type: str = "dflash"
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 131072
    # Predicted tokens per draft forward; the engine uses
    # ``K = block_size - 1`` as its drafts-per-iter.
    block_size: int = 4
    # Indices (into the TARGET model's ``layers``) whose post-layer
    # hidden states are concatenated and fed to the draft as
    # ``target_hidden_states``.
    target_layer_ids: Tuple[int, ...] = ()
    # Per-layer attention type ("full_attention" or "sliding_attention").
    # Defaults to all-full when empty.
    layer_types: Tuple[str, ...] = field(default_factory=tuple)
    # Sliding window for ``sliding_attention`` layers (None otherwise).
    sliding_window: Optional[int] = None
    mask_token_id: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    final_logit_softcapping: Optional[float] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        """Hand-roll ``from_dict`` to absorb a few schema quirks.

        Specifically:
        * The upstream checkpoint nests ``target_layer_ids`` and
          ``mask_token_id`` under a ``dflash_config`` sub-object, and
          ``rope_theta`` under ``rope_parameters``.
        * ``layer_types`` is sometimes absent (default to all-full).
        * ``head_dim`` defaults to ``hidden_size // num_attention_heads``.
        """
        params = dict(params)
        sub = params.pop("dflash_config", {}) or {}
        rope = params.get("rope_parameters")
        if rope is not None:
            params.setdefault("rope_theta", rope.get("rope_theta", cls.rope_theta))
        if "target_layer_ids" not in params and "target_layer_ids" in sub:
            params["target_layer_ids"] = tuple(sub["target_layer_ids"])
        if "mask_token_id" not in params and "mask_token_id" in sub:
            params["mask_token_id"] = int(sub["mask_token_id"])
        if "block_size" not in params and "block_size" in sub:
            params["block_size"] = int(sub["block_size"])

        if "head_dim" not in params and "hidden_size" in params:
            params["head_dim"] = (
                params["hidden_size"] // params["num_attention_heads"]
            )

        n_layers = int(params["num_hidden_layers"])
        layer_types = params.get("layer_types") or ["full_attention"] * n_layers
        if len(layer_types) != n_layers:
            raise ValueError(
                f"layer_types length {len(layer_types)} != "
                f"num_hidden_layers {n_layers}"
            )
        unknown = set(layer_types) - {"full_attention", "sliding_attention"}
        if unknown:
            raise ValueError(
                f"Unsupported DFlash layer_types: {sorted(unknown)}."
            )
        if (
            "sliding_attention" in layer_types
            and params.get("sliding_window") is None
        ):
            raise ValueError(
                "DFlash config must define sliding_window when any layer "
                "is sliding_attention."
            )
        params["layer_types"] = tuple(layer_types)
        params["target_layer_ids"] = tuple(params.get("target_layer_ids", ()))
        if not params["target_layer_ids"]:
            raise ValueError(
                "DFlash config must define target_layer_ids "
                "(either at the top level or under dflash_config)."
            )
        return super().from_dict(params)

    @property
    def num_capture_layers(self) -> int:
        return len(self.target_layer_ids)


# ════════════════════════════════════════════════════════════════════
# Attention (block-diffusion)
# ════════════════════════════════════════════════════════════════════


class DFlashAttention(nn.Module):
    """Per-layer block-diffusion attention.

    Two entry points:

    * :meth:`calibrate` — fed by the engine after a prefill / verify
      pass.  Projects ``h_ctx`` through ``k_proj`` / ``v_proj``,
      applies RoPE and writes the K/V into the draft cache at
      ``out_loc``.  No attention computation.
    * :meth:`__call__`  — the proposal forward.  Writes proposal
      K/V at ``batch.out_loc`` and runs attention via
      :meth:`AttnBackend.forward`, passing
      ``is_cross_attention`` / ``sliding_window_size`` to select the
      DFlash mask variant (see
      ``docs/dflash_kernel_requirements.md`` for the kernel
      contract).

    Per the DFlash paper / reference, K/V for proposal and context
    share the SAME ``k_proj`` / ``v_proj`` weights — they live in
    different "logical sources" (target hidden vs masked block
    embeddings) but go through identical projections.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.layer_id = layer_idx
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        layer_type = (
            args.layer_types[layer_idx]
            if args.layer_types
            else "full_attention"
        )
        self.is_sliding = layer_type == "sliding_attention"
        self.sliding_window = (
            args.sliding_window if self.is_sliding else None
        )

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rotary_dim = int(self.head_dim * args.partial_rotary_factor)
        self.rope = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            base=args.rope_theta,
        )

    def calibrate(
        self,
        h_ctx: mx.array,
        out_loc: mx.array,
        positions: mx.array,
    ) -> None:
        """Project + RoPE + store this layer's K/V from target hidden.

        Called once per draft forward (from :meth:`Model.calibrate`)
        AFTER a target pass (prefill or verify) has produced new
        target hidden states.  Writes the per-layer K/V projection
        into the draft cache at ``out_loc`` so the proposal queries
        read target-derived (not stale-proposal) K/V at those
        positions.

        Args:
            h_ctx: ``[N_ctx, hidden_size]`` — concat of ``h_ctx``
                rows across the batch, where ``h_ctx`` was produced
                by ``hidden_norm(fc(target_hidden))`` upstream.
            out_loc: ``[N_ctx]`` int32 — page IDs to write at
                (mirrored with the target's page_table entries for
                the same positions).
            positions: ``[N_ctx]`` int32 — RoPE positions in target
                sequence coordinates.
        """
        L = h_ctx.shape[0]
        k = self.k_proj(h_ctx).reshape(L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(h_ctx).reshape(L, self.n_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = self.rope(k, positions)
        get_global_ctx().attn_backend.store_kv_only(
            k, v, out_loc, self.layer_id,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Proposal forward — writes proposal K/V + runs attention.

        Reads ``ctx.batch.attn_metadata`` for positions / page IDs.
        Writes proposal K/V to ``ctx.batch.out_loc`` and runs paged
        attention via :meth:`AttnBackend.forward`, passing the
        block-diffusion mask flags:

        * ``full_attention`` layer  →  ``is_cross_attention=True,
          sliding_window_size=0`` (bidirectional within the proposal
          block, full visibility over cached prefix).
        * ``sliding_attention`` layer →  ``is_cross_attention=False,
          sliding_window_size=W`` (causal + windowed).

        See :meth:`AttnBackend.forward` for the mask truth table.
        """
        L = x.shape[0]

        q = self.q_proj(x).reshape(L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(L, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        ctx = get_global_ctx()
        metadata = ctx.batch.attn_metadata
        q = self.rope(q, metadata.positions)
        k = self.rope(k, metadata.positions)

        out = ctx.attn_backend.forward(
            q, k, v, self.layer_id, ctx.batch,
            is_cross_attention=not self.is_sliding,
            sliding_window_size=self.sliding_window or 0,
        )
        return self.o_proj(out.reshape(L, -1))


# ════════════════════════════════════════════════════════════════════
# Decoder layer
# ════════════════════════════════════════════════════════════════════


class DFlashDecoderLayer(nn.Module):
    """Standard pre-norm transformer block built on DFlashAttention."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DFlashAttention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps,
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


# ════════════════════════════════════════════════════════════════════
# Draft model (exposed as ``Model`` for mlx_serve.models.load_model)
# ════════════════════════════════════════════════════════════════════


class Model(nn.Module):
    """Multi-layer block-diffusion draft for Qwen3.5-series targets.

    Invoked twice per spec-decoding iter by :class:`DflashEngine`:

    1. **Proposal forward** (this :meth:`__call__`).  Embed the
       ``[B*(K+1)]`` block input and run all layers.  Each layer's
       :meth:`DFlashAttention.__call__` queries the cache plus its
       own proposal K/V and produces ``[B*(K+1), hidden_size]``
       post-layer hidden.  Final ``lm_head(norm(h))`` gives draft
       logits.

    2. **Calibrate** (:meth:`calibrate`).  Issued AFTER each target
       pass (prefill or decode-iter verify) over the *just-committed
       positions*: target hidden is projected through each draft
       layer's ``k_proj`` / ``v_proj`` and written to the draft
       cache at the matching page IDs.  This keeps the draft cache
       in lockstep with the target cache at every iter boundary.

    The split exists because the proposal queries and the
    calibration writes have *different output cardinalities* — the
    proposal has ``B*(K+1)`` queries while the calibration just
    appends K/V — and the current kernel API expects ``len(Q) ==
    len(K/V writes)``.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        concat_dim = args.num_capture_layers * args.hidden_size
        self.fc = nn.Linear(concat_dim, args.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps,
        )

        self.layers = [
            DFlashDecoderLayer(args, i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Populated by :func:`load_dflash_draft_model` from the
        # already-loaded target model.
        self.embed_tokens: nn.Embedding | None = None
        self.lm_head: Any = None

    @property
    def num_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def num_kv_heads(self) -> int:
        return self.args.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.args.head_dim

    # ── Weight loading hook (called by load_model) ────────────────────

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Drop embed_tokens / lm_head — bound separately from target.

        Match the reference DFlash checkpoint layout (no ``model.``
        prefix); module attribute names are
        ``fc / hidden_norm / layers / norm`` which lines up 1:1 with
        ``fc.weight / hidden_norm.weight / layers.{i}.* / norm.weight``.
        """
        weights.pop("embed_tokens.weight", None)
        weights.pop("lm_head.weight", None)
        return weights

    # ── Calibration write (driven by the engine) ──────────────────────

    def calibrate(
        self,
        target_hidden: mx.array,
        out_loc: mx.array,
        positions: mx.array,
    ) -> None:
        """Write per-layer projection of target hidden to the draft cache.

        Args:
            target_hidden: ``[N_ctx, num_capture_layers * hidden_size]``
                — concat'd target hidden over the capture layers,
                concatenated across the batch in req order.
            out_loc: ``[N_ctx]`` int32 — page IDs for the calibration
                writes (= page_table entries for the same positions
                in the target sequence, mirrored layout).
            positions: ``[N_ctx]`` int32 — RoPE positions in target
                sequence coordinates.

        No-op when ``target_hidden`` is empty.
        """
        if target_hidden.shape[0] == 0:
            return
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer in self.layers:
            layer.self_attn.calibrate(h_ctx, out_loc, positions)

    # ── Proposal forward ──────────────────────────────────────────────

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Proposal forward over a ``[B*(K+1)]`` block input.

        Returns:
            ``[B*(K+1), vocab_size]`` raw logits.  Callers typically
            extract per-req slices ``[1..K]`` (the K useful drafts)
            via the qo_indptr in ``ctx.batch.attn_metadata``.
        """
        assert self.embed_tokens is not None and self.lm_head is not None
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        if self.args.final_logit_softcapping is not None:
            cap = self.args.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits


# ════════════════════════════════════════════════════════════════════
# Loader
# ════════════════════════════════════════════════════════════════════


def load_dflash_draft_model(
    dflash_path: str,
    target_model: nn.Module,
) -> "Model":
    """Load a DFlash draft, sharing embed / lm_head with *target_model*.

    Delegates the bulk of the work — config parsing, weight loading
    and quantisation — to the generic
    :func:`mlx_serve.models.load_model`; the only extra step is
    forcing ``model_type="dflash"`` (so the discovery picks up this
    module) and binding shared weights from the target.

    Args:
        dflash_path: Local directory or HF repo containing the DFlash
            ``config.json`` + ``*.safetensors``.
        target_model: An already-loaded mlx-serve target model; its
            ``model.embed_tokens`` (and ``lm_head`` if not tied) is
            shared with the draft, mirroring the reference impl.

    Returns:
        The fully-initialised :class:`Model` with weights loaded,
        evaluated, and shared embeddings bound.  Config is
        accessible as ``draft.args``.
    """
    from mlx_serve.models import load_model

    draft, _ = load_model(
        dflash_path,
        model_config={"model_type": "dflash"},
    )

    target_args = target_model.args
    draft.embed_tokens = target_model.model.embed_tokens
    if getattr(target_args, "tie_word_embeddings", False):
        draft.lm_head = target_model.model.embed_tokens.as_linear
    else:
        draft.lm_head = target_model.lm_head
    return draft
