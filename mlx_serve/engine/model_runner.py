import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from mlx_serve.models.qwen3 import Qwen3ForCausalLM, ModelArgs
from mlx_serve.layers.logits_processor import LogitsProcessor
from mlx_serve.layers.sampler import TopKTopPSampler
from mlx_serve.engine.forward_batch import ForwardBatch, ForwardType
from mlx_serve.mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
from mlx_serve.mem_cache.allocator import TokenToKVPoolAllocator
from mlx_serve.mem_cache.radix_cache import RadixCache
from mlx_serve.utils import load_model


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        radix_cache: RadixCache,
        max_num_batched_tokens: int,
        max_model_len: int,
    ):
        self.model, self.model_args = load_model(model_path)
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.radix_cache = radix_cache
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.logits_processor = LogitsProcessor()
        self.sampler = TopKTopPSampler()
        
        # Warmup model
        self.warmup_model()

    def warmup_model(self):
        """Warmup the model with dummy inputs."""
        input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        forward_batch = ForwardBatch(
            request_ids=[0],
            seq_lens=[5],
            offsets=[0],
            forward_type=ForwardType.prefill,
            temperatures=[1.0],
        )
        self.model(input_ids, forward_batch)

    def run(
        self,
        forward_batch: ForwardBatch,
        input_ids: mx.array,
    ) -> Optional[mx.array]:
        """
        Run the model forward pass on a prepared batch.
        
        Args:
            forward_batch: The ForwardBatch prepared by scheduler
            input_ids: The input token IDs array
            
        Returns:
            Sampled token IDs array, or None if batch is invalid
        """
        if forward_batch is None or input_ids is None:
            return None
        
        # Forward pass through model
        hidden_states = self.model.model(input_ids, forward_batch=forward_batch)
        
        # Process logits (apply LM head and get logprobs for last positions)
        logits = self.logits_processor(
            hidden_states, 
            forward_batch, 
            self.model, 
            self.model_args.tie_word_embeddings
        )
        
        # Sample tokens
        token_ids = self.sampler(logits, forward_batch)
        return token_ids
