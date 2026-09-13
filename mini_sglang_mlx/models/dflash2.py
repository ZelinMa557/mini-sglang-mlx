"""DFlash2 (block-diffusion) draft model.

DFlash2 keeps the DFlash backbone — a small multi-layer draft that fills
in ``block_size - 1`` masked slots per forward from the target's captured
hidden states — and adds two cheap modules on top, each aimed at one of
the two ways an independently-predicted block loses accepted tokens:

* **Candidate selector** (coherence).  Independent per-position top-1
  picks are individually plausible but need not fit together, so the
  block gets cut short at verification.  The selector keeps the top
  ``selector_top_k`` candidates at every position and scores each
  adjacent ``(predecessor, candidate)`` pair with a low-rank bilinear
  term, then walks the best path.  Scoring is fully parallel; only the
  walk is sequential.

* **Block-local dynamic convolution** (suffix decay).  Recall falls
  toward the end of a block because the candidates themselves run out.
  A two-tap grouped depthwise conv, whose coefficients combine a learned
  base kernel with a correction computed from the current hidden state,
  is inserted before and after each attention and MLP sublayer.  It is
  block-local and stateless — position ``l`` reads position ``l - 1`` —
  so it stays parallel while letting information cross the block.

Everything else (``fc`` projection of the target hidden, shared
embed / LM head, calibration of the draft KV cache, mirrored page
layout) is inherited from :mod:`mini_sglang_mlx.models.dflash` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn

from mini_sglang_mlx.core import get_global_ctx

from . import dflash


# ════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════


@dataclass
class ModelArgs(dflash.ModelArgs):
    """DFlash2 draft config — DFlash's fields plus the two new modules."""

    model_type: str = "dflash2"
    # Two-tap conv straddling each attention / MLP sublayer.
    conv_kernel_size: int = 2
    # Channels sharing one dynamic conv correction.
    conv_group_size: int = 16
    # Rank of the selector's bilinear token codebooks.
    selector_rank: int = 256
    # Candidate shortlist size at every block position.
    selector_top_k: int = 16

    @classmethod
    def from_dict(cls, params: Dict) -> "ModelArgs":
        """Lift the DFlash2 knobs out of the ``dflash_config`` sub-object.

        Upstream ships them nested (alongside ``block_size`` and
        ``target_layer_ids``, which the parent already handles).
        """
        params = dict(params)
        sub = params.get("dflash_config") or {}
        for key in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            if key not in params and key in sub:
                params[key] = int(sub[key])
        return super().from_dict(params)


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════


def _block_shape(num_tokens: int) -> Tuple[int, int]:
    """Resolve ``(batch, block_len)`` for flat ``[B * L, hidden]`` activations.

    The proposal forward packs every req's ``block_size`` proposal
    tokens into one flat tensor, so the block structure can only come
    from the batch the draft context is currently bound to.
    """
    B = len(get_global_ctx().batch.padded_reqs)
    assert B > 0 and num_tokens % B == 0, (
        f"Cannot infer a uniform block length for {num_tokens} tokens "
        f"across {B} reqs."
    )
    return B, num_tokens // B


def _grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> mx.array:
    """Two-tap grouped dynamic depthwise conv over the block axis.

    Args:
        hidden: ``[B, L, hidden_size]``.
        dynamic: ``[B, L, kernel_size, groups]`` per-token corrections.
        base: ``[kernel_size, hidden_size]`` learned static kernel.

    Position ``l`` reads positions ``l - offset`` (zeros before the block
    start), so information flows across the block while every position is
    still computed in parallel.

    The accumulation order mirrors the reference implementation exactly
    (per tap: base term then dynamic term, both in the activations'
    dtype) so the draft backbone stays numerically identical to the
    checkpoint's training-time arithmetic.
    """
    B, L, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(B, L, groups, group_size)

    output = mx.zeros_like(blocks)
    for offset in range(base.shape[0]):
        if offset == 0:
            values = blocks
        else:
            values = mx.concatenate(
                [mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]],
                axis=1,
            )
        kernel = base[offset].reshape(
            1, 1, groups, group_size,
        ).astype(hidden.dtype)
        output = output + kernel * values
        output = output + dynamic[:, :, offset][..., None] * values
    return output.reshape(B, L, hidden_size)


# ════════════════════════════════════════════════════════════════════
# Modules
# ════════════════════════════════════════════════════════════════════


class GroupedDynamicCausalConv(nn.Module):
    """Dynamic conv straddling one sublayer (attention or MLP).

    :meth:`prepare` runs on the sublayer *input* and applies the
    input-side tap, returning a per-token correction that :meth:`finish`
    applies to the sublayer *output*.  Both ends see the same hidden
    state's projection, so one ``kernel_projection`` feeds both taps.
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        # Leading dim 2 = {prepare tap, finish tap}.
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False,
        )

    def _as_block(self, hidden: mx.array) -> mx.array:
        B, L = _block_shape(hidden.shape[0])
        return hidden.reshape(B, L, -1)

    def prepare(self, hidden: mx.array) -> Tuple[mx.array, mx.array]:
        """Apply the input-side tap; returns ``(output, carry)``."""
        blocks = self._as_block(hidden)
        groups = blocks.shape[-1] // self.group_size
        dynamic = self.kernel_projection(blocks).reshape(
            *blocks.shape[:-1], 2, self.kernel_size, groups,
        )
        out = _grouped_dynamic_convolve(
            blocks, dynamic[:, :, 0], self.base_kernel[0], self.group_size,
        )
        return out.reshape(hidden.shape), dynamic[:, :, 1]

    def finish(self, hidden: mx.array, carry: mx.array) -> mx.array:
        """Apply the output-side tap using the carry from :meth:`prepare`."""
        out = _grouped_dynamic_convolve(
            self._as_block(hidden), carry, self.base_kernel[1], self.group_size,
        )
        return out.reshape(hidden.shape)


class CandidateSelector(nn.Module):
    """Adjacent-pair path selector over the draft's top-k candidates.

    At block position ``t`` with predecessor ``a`` and candidate ``b``::

        S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>

    ``U_t(b)`` is the draft's own logit for ``b`` (independent
    plausibility), ``A`` / ``B`` are the predecessor / successor token
    codebooks, and ``H(h_t)`` is a context gate over the hidden state
    deciding which parts of the match count.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.selector_top_k
        self.predecessor_codebook = nn.Embedding(
            args.vocab_size, args.selector_rank,
        )
        self.successor_codebook = nn.Embedding(
            args.vocab_size, args.selector_rank,
        )
        self.hidden_projection = nn.Linear(
            args.hidden_size, args.selector_rank, bias=False,
        )

    def select(
        self,
        hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
    ) -> mx.array:
        """Walk the best path through the candidate graph (greedy).

        Args:
            hidden: ``[B, K, hidden_size]`` post-``norm`` draft hidden.
            logits: ``[B, K, vocab_size]`` draft logits at the same slots.
            anchor_ids: ``[B]`` int32 — the already-committed token each
                walk starts from (the block's slot 0).

        Returns:
            ``[B, K]`` int32 — one selected token per block slot.

        Only the temperature-0 walk is implemented, matching the
        greedy-only verification supported by the engine.  The upstream
        sampling variant additionally draws from ``S`` and returns its
        probabilities for rejection sampling.
        """
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[
            ..., -self.top_k:
        ]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)

        predecessor = anchor_ids
        path = []
        for position in range(hidden.shape[1]):
            edges = mx.sum(
                self.predecessor_codebook(predecessor)[:, None]
                * hidden[:, position, None]
                * self.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            selected = mx.argmax(unary[:, position] + edges, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1,
            )[:, 0]
            path.append(predecessor)
        return mx.stack(path, axis=1)


class DFlash2DecoderLayer(dflash.DFlashDecoderLayer):
    """DFlash decoder layer with a dynamic conv around each sublayer."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.attention_conv = GroupedDynamicCausalConv(
            args.hidden_size, args.conv_kernel_size, args.conv_group_size,
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            args.hidden_size, args.conv_kernel_size, args.conv_group_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x, carry = self.attention_conv.prepare(self.input_layernorm(x))
        x = residual + self.attention_conv.finish(self.self_attn(x), carry)
        residual = x
        x, carry = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return residual + self.mlp_conv.finish(self.mlp(x), carry)


# ════════════════════════════════════════════════════════════════════
# Draft model (exposed as ``Model`` for mini_sglang_mlx.models.load_model)
# ════════════════════════════════════════════════════════════════════


class Model(dflash.Model):
    """DFlash2 draft: DFlash backbone + selector + block-local conv."""

    layer_class = DFlash2DecoderLayer

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.candidate_selector = CandidateSelector(args)

    # ── Weight loading hook (called by load_model) ────────────────────

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Rename the selector codebooks to their ``nn.Embedding`` keys.

        The checkpoint stores them as bare embedding tables
        (``candidate_selector.predecessor_codebook``) while
        :class:`mlx.nn.Embedding` exposes a ``.weight`` member.
        """
        for name in ("predecessor_codebook", "successor_codebook"):
            key = f"candidate_selector.{name}"
            if key in weights:
                weights[f"{key}.weight"] = weights.pop(key)
        return super().sanitize(weights)

    # ── Proposal forward ──────────────────────────────────────────────

    def propose(self, input_ids: mx.array) -> mx.array:
        """Select the draft tokens for every req's block.

        The block is ``[T, MASK, ..., MASK]`` of length ``K + 1``: slot 0
        is the pending token ``T``, slots ``1..K`` are the masked
        positions the draft fills in.  Only the latter are scored, and
        the walk starts from ``T``.

        Returns:
            ``[B, K]`` int32 draft tokens.
        """
        B, L = _block_shape(input_ids.shape[0])
        hidden = self.hidden_states(input_ids).reshape(B, L, -1)
        return self.candidate_selector.select(
            hidden[:, 1:],
            self.compute_logits(hidden)[:, 1:],
            input_ids.reshape(B, L)[:, 0],
        )
