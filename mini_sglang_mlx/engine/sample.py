from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, List

import mlx.core as mx

if TYPE_CHECKING:
    from mini_sglang_mlx.core import Batch

MIN_P = 1e-6
MIN_T = 1e-6


@dataclass
class BatchSamplingArgs:
    temperatures: mx.array | None
    top_p: mx.array | None = None

@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _apply_top_p(logprobs: mx.array, top_ps: mx.array) -> mx.array:
    """Mask logprobs to nucleus (top-p) per row. Shape: [batch, vocab], top_ps: [batch]."""
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    vocab_size = logprobs.shape[-1]
    vocab_indices = mx.arange(vocab_size)[None, :]
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices), sorted_indices, vocab_indices, axis=-1
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)
    return mx.where(
        cumulative_probs > 1 - top_ps[:, None],
        logprobs,
        -float("inf"),
    )


def _sample(logits: mx.array, args: BatchSamplingArgs) -> mx.array:
    """Sample next tokens: greedy or temperature + optional top_p."""
    if args.temperatures is None:
        return mx.argmax(logits, axis=-1)

    temps = args.temperatures
    zero_temp = temps == 0
    if mx.all(zero_temp).item():
        return mx.argmax(logits, axis=-1)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    if args.top_p is not None:
        logprobs = _apply_top_p(logprobs, args.top_p)

    scaled = logprobs / mx.maximum(temps[:, None], MIN_T)
    sampled = mx.random.categorical(scaled)
    greedy = mx.argmax(logits, axis=-1)
    return mx.where(zero_temp, greedy, sampled)


@dataclass
class Sampler:
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        ts = [
            max(0.0 if p.is_greedy else p.temperature, MIN_T)
            for p in params
        ]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]

        temperatures = mx.array(ts, dtype=mx.float32)
        top_p = None
        if any(p < 1.0 for p in top_ps):
            top_p = mx.array(top_ps, dtype=mx.float32)
        return BatchSamplingArgs(temperatures, top_p=top_p)

    def sample(self, logits: mx.array, args: BatchSamplingArgs) -> mx.array:
        return _sample(logits, args).astype(mx.int32)
