from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Tuple

import mlx.core as mx

from mlx_serve.attention import AttnBackend
from mlx_serve.core import Batch, Context, Req, set_global_ctx
from mlx_serve.kvcache.mha_pool import MHAKVCache
from mlx_serve.models import create_model
from mlx_serve.utils import init_logger

from .config import EngineConfig
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens: mx.array


def create_page_table(shape: Tuple[int, int]) -> mx.array:
    return mx.zeros(shape, dtype=mx.int32)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


@dataclass(frozen=True)
class _ModelMeta:
    head_dim: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int
    max_position: int

    @staticmethod
    def from_hf_config(hf_config: Dict[str, Any]) -> "_ModelMeta":
        head_dim = int(
            hf_config.get("head_dim")
            or hf_config["hidden_size"] // hf_config["num_attention_heads"]
        )
        return _ModelMeta(
            head_dim=head_dim,
            num_kv_heads=int(
                hf_config.get("num_key_value_heads", hf_config["num_attention_heads"])
            ),
            num_layers=int(hf_config.get("num_hidden_layers", hf_config.get("num_layers"))),
            vocab_size=int(hf_config["vocab_size"]),
            max_position=int(hf_config.get("max_position_embeddings", 8192)),
        )


class Engine:
    def __init__(self, config: EngineConfig):
        self.dtype = config.dtype

        self.model, hf_config = create_model(config.model_path, lazy=False)
        self.model_meta = _ModelMeta.from_hf_config(hf_config)
        self.num_pages = self.dummy_page = self._determine_num_pages(config)
        self.kv_cache = MHAKVCache(
            num_kv_heads=self.model_meta.num_kv_heads,
            num_layers=self.model_meta.num_layers,
            head_dim=self.model_meta.head_dim,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            dtype=self.dtype,
        )

        max_seq_len = (
            config.max_seq_len_override
            if config.max_seq_len_override is not None
            else self.model_meta.max_position
        )
        self.max_seq_len = _align_up_32(min(max_seq_len, self.num_pages))
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
        )
        self.attn_backend = AttnBackend(
            config=self.model_meta,  # type: ignore[arg-type]
            kvcache=self.kv_cache,
            page_table=self.page_table,
        )
        self.ctx = Context(page_size=1, attn_backend=self.attn_backend)
        set_global_ctx(self.ctx)
        self.sampler = Sampler(self.model_meta.vocab_size)

        self.dummy_req = Req(
            input_ids=mx.array([0], dtype=mx.int32),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx, :] = self.dummy_page

    def _determine_num_pages(self, config: EngineConfig) -> int:
        cache_per_page = (
            2  # key + value
            * self.model_meta.head_dim
            * self.model_meta.num_kv_heads
            * config.page_size
            * 2 # 2 == sizeof(float16) == sizeof(bfloat16)
            * self.model_meta.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            # Conservative default for Apple unified memory. Override with
            # --num-pages when precise control is needed.
            max_seq_len = (
                config.max_seq_len_override
                if config.max_seq_len_override is not None
                else self.model_meta.max_position
            )
            num_pages = min(max_seq_len * max(config.max_running_req, 1), 262_144)

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        num_pages = 8192 * 4 # todo
        real_kv_size = num_pages * cache_per_page / (1024**3)
        logger.info("Allocating %s pages for KV cache, K + V = %.2f GB", num_pages, real_kv_size)
        return num_pages

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        with self.ctx.forward_batch(batch):
            logits = self.model()

        last_indices = batch.attn_metadata.get_last_indices(batch.size)
        last_logits = logits[last_indices]

        for req in batch.reqs:
            req.complete_one()

        next_tokens = self.sampler.sample(last_logits, args)
        return ForwardOutput(next_tokens=mx.astype(next_tokens, mx.int32))

    def shutdown(self) -> None:
        pass