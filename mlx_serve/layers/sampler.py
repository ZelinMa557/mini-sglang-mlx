from functools import partial
import mlx.core as mx
import mlx.nn as nn
from mlx_serve.engine.forward_batch import ForwardBatch

@mx.compile(inputs=mx.random.state, outputs=mx.random.state)
def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logprobs: A vector of log probabilities.
        top_p: The cumulative probability threshold for top-p filtering.
    Returns:
        token selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.exp(logprobs)
    # sort in ascending order
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Rearrange cumulative probs back to original order
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    # select tokens with cumulative probs below threshold
    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )

@mx.compile(inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k(
    logprobs: mx.array,
    top_k: int,
) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability.

    Args:
        logprobs: A vector of log probabilities.
        top_k (int): Top k tokens to sample from.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return masked_logprobs

@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))

class TopKTopPSampler(nn.Module):
    def __init__(
            self,
            temperature: float,
            top_k: int,
            top_p: float
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def __call__(self, logits: mx.array, forward_batch: ForwardBatch) -> mx.array:
        logits = apply_top_k(logits, self.top_k)
        logits = apply_top_p(logits, self.top_p)
        if self.temperature == 0:
            return mx.argmax(logits, axis=-1)
        return categorical_sampling(logits, self.temperature)