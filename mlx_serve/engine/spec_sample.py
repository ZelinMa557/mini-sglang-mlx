"""Verification helpers for speculative decoding (greedy / T=0 only).

This module is intentionally minimal: it only supports temperature-0
verification, which collapses to "target argmax must equal draft token
to accept". A future sampling-based version would live alongside this.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx


class GreedyVerifyResult(NamedTuple):
    """Outputs of :func:`greedy_verify`.

    All tensors live on device and are batched along the first axis.
    """

    # ``[B]`` int32 — number of *drafts* accepted per req, in ``[0, K]``.
    # The total tokens committed to the target KV this iter equals
    # ``num_drafts_accepted + 1`` (the +1 is the previously-pending ``T``,
    # which is always accepted because target already sampled it).
    num_drafts_accepted: mx.array
    # ``[B]`` int32 — the next pending token for each req. Equal to
    # ``target_preds[i, num_drafts_accepted[i]]`` — i.e. either the
    # corrected token at the first mismatch or, when all K drafts are
    # accepted, the bonus token from the last verify position.
    bonus_tokens: mx.array
    # ``[B, K+1]`` int32 — target's argmax at every verify position;
    # retained for callers that need to splice accepted tokens into
    # ``input_ids`` without an extra Python loop.
    target_preds: mx.array


def greedy_verify(
    verify_logits: mx.array,
    drafts: mx.array,
) -> GreedyVerifyResult:
    """Greedy MTP/EAGLE verification.

    Args:
        verify_logits: ``[B, K+1, V]`` — target's logits at every verify
            position. Position 0 is the previously-pending ``T``, positions
            1..K are the K draft tokens.
        drafts: ``[B, K]`` int32 — the K draft tokens sampled by the
            draft model in this iter.

    Returns:
        A :class:`GreedyVerifyResult` with batched ``num_drafts_accepted``,
        ``bonus_tokens`` and ``target_preds``.
    """
    B, K = drafts.shape
    assert verify_logits.shape[0] == B and verify_logits.shape[1] == K + 1, (
        f"verify_logits shape {verify_logits.shape} does not match "
        f"drafts shape {drafts.shape}"
    )

    target_preds = mx.argmax(verify_logits, axis=-1).astype(mx.int32)  # [B, K+1]

    # matches[i, k] = 1 iff target_preds[i, k] == drafts[i, k]; cumprod
    # along K stays 1 until the first mismatch, then collapses to 0.
    matches = (target_preds[:, :K] == drafts).astype(mx.int32)  # [B, K]
    accept_mask = mx.cumprod(matches, axis=1)  # [B, K]
    num_drafts_accepted = mx.sum(accept_mask, axis=1).astype(mx.int32)  # [B]

    # ``bonus`` is the corrected token at the first rejected position, or
    # the genuine bonus from position K when all drafts are accepted.
    bonus_tokens = mx.take_along_axis(
        target_preds, num_drafts_accepted[:, None], axis=1,
    ).squeeze(axis=1)  # [B]

    return GreedyVerifyResult(
        num_drafts_accepted=num_drafts_accepted,
        bonus_tokens=bonus_tokens,
        target_preds=target_preds,
    )


def gather_last_accepted_hidden(
    verify_hidden: mx.array,
    num_drafts_accepted: mx.array,
) -> mx.array:
    """Pick the target hidden state at each req's last accepted position.

    Args:
        verify_hidden: ``[B, K+1, D]`` target hidden states from the
            verify forward.
        num_drafts_accepted: ``[B]`` from :func:`greedy_verify`.

    Returns:
        ``[B, D]`` — the hidden state at index ``num_drafts_accepted[i]``
        for each req. Used as ``target_hidden_states`` for the bonus
        draft step that produces ``pending_draft_*`` for the next iter.
    """
    idx = num_drafts_accepted[:, None, None]
    # Broadcasting take_along_axis with a 3-D index along axis 1 then
    # squeeze — keeps the whole gather on-device.
    return mx.take_along_axis(
        verify_hidden,
        mx.broadcast_to(idx, (verify_hidden.shape[0], 1, verify_hidden.shape[-1])),
        axis=1,
    ).squeeze(axis=1)
