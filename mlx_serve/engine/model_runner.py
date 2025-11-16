import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional
from mlx_serve.models.qwen3 import Qwen3ForCausalLM, ModelArgs
from mlx_serve.layers.logits_processor import LogitsProcessor
from mlx_serve.layers.sampler import TopKTopPSampler
from mlx_serve.engine.forward_batch import ForwardBatch, ForwardType
from mlx_serve.engine.forward_request import ForwardRequest
from mlx_serve.mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
from mlx_serve.mem_cache.allocator import TokenToKVPoolAllocator
from mlx_serve.mem_cache.radix_cache import RadixCache


class ModelRunner:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        model_args: ModelArgs,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        radix_cache: RadixCache,
        max_num_batched_tokens: int,
        max_model_len: int,
    ):
        self.model = model
        self.model_args = model_args
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

    def prepare_prefill_batch(
        self,
        requests: List[ForwardRequest],
        prefix_indices_list: List[mx.array],
    ) -> ForwardBatch:
        """Prepare a batch for prefill phase."""
        input_ids_list = []
        seq_lens = []
        offsets = []
        request_ids = []
        temperatures = []
        top_ks = []
        top_ps = []
        current_offset = 0
        
        for req, prefix_indices in zip(requests, prefix_indices_list):
            # Get all tokens (input + generated so far)
            all_tokens = (req.input_tokens or []) + (req.generated_tokens or [])
            
            # Calculate how many tokens need to be processed
            # prefix_indices contains the matched prefix from radix cache
            prefix_len = len(prefix_indices) if prefix_indices is not None else 0
            tokens_to_process = all_tokens[prefix_len:]
            
            if len(tokens_to_process) == 0:
                continue
                
            input_ids_list.extend(tokens_to_process)
            seq_len = len(tokens_to_process)
            seq_lens.append(seq_len)
            offsets.append(current_offset)
            current_offset += seq_len
            request_ids.append(req.id)
            temperatures.append(req.temperature if req.temperature is not None else 1.0)
            top_ks.append(req.top_k if req.top_k is not None else -1)
            top_ps.append(req.top_p if req.top_p is not None else 1.0)
        
        if len(input_ids_list) == 0:
            return None
            
        input_ids = mx.array(input_ids_list, dtype=mx.int32)
        batch = ForwardBatch(
            request_ids=request_ids,
            seq_lens=mx.array(seq_lens, dtype=mx.int32),
            offsets=mx.array(offsets, dtype=mx.int32),
            forward_type=ForwardType.prefill,
            temperatures=mx.array(temperatures, dtype=mx.float32),
            top_ks=mx.array(top_ks, dtype=mx.int32),
            top_ps=mx.array(top_ps, dtype=mx.float32),
        )
        batch._init_position_ids()
        return batch, input_ids

    def prepare_decode_batch(
        self,
        requests: List[ForwardRequest],
    ) -> tuple[ForwardBatch, mx.array]:
        """Prepare a batch for decode phase."""
        input_ids_list = []
        seq_lens = []
        offsets = []
        request_ids = []
        temperatures = []
        top_ks = []
        top_ps = []
        current_offset = 0
        
        for req in requests:
            # For decode, we only process the last token
            all_tokens = (req.input_tokens or []) + (req.generated_tokens or [])
            if len(all_tokens) == 0:
                continue
                
            last_token = all_tokens[-1]
            input_ids_list.append(last_token)
            seq_lens.append(1)
            offsets.append(current_offset)
            current_offset += 1
            request_ids.append(req.id)
            temperatures.append(req.temperature if req.temperature is not None else 1.0)
            top_ks.append(req.top_k if req.top_k is not None else -1)
            top_ps.append(req.top_p if req.top_p is not None else 1.0)
        
        if len(input_ids_list) == 0:
            return None, None
            
        input_ids = mx.array(input_ids_list, dtype=mx.int32)
        batch = ForwardBatch(
            request_ids=request_ids,
            seq_lens=mx.array(seq_lens, dtype=mx.int32),
            offsets=mx.array(offsets, dtype=mx.int32),
            forward_type=ForwardType.decode,
            temperatures=mx.array(temperatures, dtype=mx.float32),
            top_ks=mx.array(top_ks, dtype=mx.int32),
            top_ps=mx.array(top_ps, dtype=mx.float32),
        )
        return batch, input_ids

    def run(
        self,
        requests: List[ForwardRequest],
        prefix_indices_list: List[Optional[mx.array]],
        is_prefill: bool,
    ) -> mx.array:
        """Run the model forward pass and return sampled tokens."""
        if is_prefill:
            batch, input_ids = self.prepare_prefill_batch(requests, prefix_indices_list)
        else:
            batch, input_ids = self.prepare_decode_batch(requests)
            
        if batch is None or input_ids is None:
            return None
            
        # Forward pass
        # Call model.model directly since Qwen3Model accepts forward_batch
        hidden_states = self.model.model(input_ids, forward_batch=batch)
        
        # Apply LM head
        logits = self.logits_processor(hidden_states, batch, self.model, self.model_args.tie_word_embeddings)
        # Sample tokens
        token_ids = self.sampler(logits, batch)
        return token_ids
