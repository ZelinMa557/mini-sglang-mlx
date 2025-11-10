from functools import partial
import mlx.core as mx
import mlx.nn as nn
from mlx_serve.engine.forward_batch import ForwardBatch

@mx.compile(inputs=mx.random.state, outputs=mx.random.state)
def apply_top_p(logprobs: mx.array, top_ps: mx.array) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits for each request in the batch.

    Args:
        logprobs: A batch of log probabilities, shape [batch_size, vocab_size].
        top_ps: Top-p values for each request, shape [batch_size].
    Returns:
        Masked logprobs with top-p filtering applied per request.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    batch_size, vocab_size = logprobs.shape
    probs = mx.exp(logprobs)
    
    # sort in ascending order for each request
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # compute cumulative probabilities for each request
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Rearrange cumulative probs back to original order
    vocab_indices = mx.arange(vocab_size)[None, :]
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        vocab_indices,
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    # expand top_ps to match logprobs shape for broadcasting
    top_ps_expanded = top_ps[:, None]  # [batch_size, 1]
    
    # select tokens with cumulative probs below threshold for each request
    return mx.where(
        cumulative_probs > 1 - top_ps_expanded,
        logprobs,
        -float("inf"),
    )

@mx.compile(inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k(
    logprobs: mx.array,
    top_ks: mx.array,
) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability for each request in the batch.

    Args:
        logprobs: A batch of log probabilities, shape [batch_size, vocab_size].
        top_ks: Top k values for each request, shape [batch_size].
    Returns:
        Masked logprobs with top-k filtering applied per request.
    """
    batch_size, vocab_size = logprobs.shape
    
    # For each request, we need to mask tokens beyond its top_k
    # We'll use argpartition to find the top-k indices for each request
    # Since argpartition doesn't support different k per batch, we'll use a different approach
    
    # Sort logprobs in descending order for each request
    sorted_indices = mx.argsort(-logprobs, axis=-1)  # descending order
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)
    
    # Create a mask: for each request, keep only the first top_k tokens
    vocab_indices = mx.arange(vocab_size)[None, :]  # [1, vocab_size]
    top_ks_expanded = top_ks[:, None]  # [batch_size, 1]
    
    # Mask: True for positions < top_k for each request
    keep_mask = vocab_indices < top_ks_expanded  # [batch_size, vocab_size]
    
    # Apply mask to sorted logprobs
    masked_sorted_logprobs = mx.where(
        keep_mask,
        sorted_logprobs,
        -float("inf")
    )
    
    # Rearrange back to original order
    # Create inverse indices: for each position in sorted_indices, find where it came from
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        vocab_indices,
        axis=-1,
    )
    masked_logprobs = mx.take_along_axis(masked_sorted_logprobs, inverse_indices, axis=-1)
    
    return masked_logprobs

@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temps):
    """
    Sample from categorical distribution with per-request temperatures.
    
    Args:
        logits: A batch of logits, shape [batch_size, vocab_size].
        temps: Temperature values for each request, shape [batch_size].
    Returns:
        Sampled token indices for each request, shape [batch_size].
    """
    # Expand temps to match logits shape for broadcasting
    temps_expanded = temps[:, None]  # [batch_size, 1]
    # Apply temperature scaling per request
    scaled_logits = logits * (1 / temps_expanded)
    # Sample for each request independently
    return mx.random.categorical(scaled_logits)

class TopKTopPSampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, logits: mx.array, forward_batch: ForwardBatch) -> mx.array:
        """
        Sample tokens with per-request temperature, top_k, and top_p settings.
        
        Args:
            logits: A batch of logits, shape [batch_size, vocab_size].
            forward_batch: ForwardBatch containing temperatures, top_ks, and top_ps arrays.
        Returns:
            Sampled token indices for each request, shape [batch_size].
        """
        batch_size = logits.shape[0]
        temperatures = forward_batch.temperatures
        top_ks = forward_batch.top_ks
        top_ps = forward_batch.top_ps
        
        # Apply top-k filtering per request
        if top_ks is not None:
            logits = apply_top_k(logits, top_ks)
        
        # Apply top-p filtering per request
        if top_ps is not None:
            logits = apply_top_p(logits, top_ps)
        
        # Handle temperature-based sampling
        if temperatures is not None:
            # Check if any request has temperature == 0 (greedy decoding)
            zero_temp_mask = temperatures == 0
            if mx.all(zero_temp_mask):
                # All requests use greedy decoding
                return mx.argmax(logits, axis=-1)
            elif mx.any(zero_temp_mask):
                # Mixed: some greedy, some sampling
                # For greedy requests, use argmax; for others, use sampling
                greedy_tokens = mx.argmax(logits, axis=-1)
                sampled_tokens = categorical_sampling(logits, temperatures)
                return mx.where(zero_temp_mask, greedy_tokens, sampled_tokens)
            else:
                # All requests use sampling
                return categorical_sampling(logits, temperatures)
        else:
            # Default to greedy if no temperatures provided
            return mx.argmax(logits, axis=-1)