import mlx.core as mx
import mlx.nn as nn
from mlx_serve.engine.forward_batch import ForwardBatch

class LogitsProcessor(nn.Module):
    def __init__(
            self,
    ) -> None:
        super().__init__()

    
    def __call__(self, hidden_states: mx.array, forward_batch: ForwardBatch, model, tie_word_embeddings: bool) -> mx.array:
        hidden_states = hidden_states[forward_batch.last_positions, :]

        if tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = model.lm_head(hidden_states)
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        return logprobs