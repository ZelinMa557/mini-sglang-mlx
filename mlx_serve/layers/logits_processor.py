import mlx.core as mx
import mlx.nn as nn
from mlx_serve.engine.forward_batch import ForwardBatch

class LogitsProcessor(nn.Module):
    def __init__(
            self,
    ) -> None:
        super().__init__()

    
    def __call__(self, hidden_states: mx.array, lm_head: nn.Linear, forward_batch: ForwardBatch) -> mx.array:
        hidden_states = hidden_states[forward_batch.last_positions, :]
        logits = lm_head(hidden_states)
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        return logprobs