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
        tc = hf_config.get("text_config", hf_config)
        head_dim = int(
            tc.get("head_dim")
            or tc["hidden_size"] // tc["num_attention_heads"]
        )
        total_layers = int(tc.get("num_hidden_layers", tc.get("num_layers")))
        interval = int(tc.get("full_attention_interval", 0))
        if interval > 0:
            num_kv_layers = total_layers // interval
        else:
            num_kv_layers = total_layers
        return _ModelMeta(
            head_dim=head_dim,
            num_kv_heads=int(
                tc.get("num_key_value_heads", tc["num_attention_heads"])
            ),
            num_layers=num_kv_layers,
            vocab_size=int(tc["vocab_size"]),
            max_position=int(tc.get("max_position_embeddings", 8192)),
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

        self.is_hybrid = getattr(self.model, "is_hybrid", False)
        self.mamba_pool = None
        self.gdn_backend = None
        if self.is_hybrid:
            self.mamba_pool = self._create_mamba_pool(config)
            from mlx_serve.attention import GDNBackend
            self.gdn_backend = GDNBackend(self.mamba_pool)

        self.ctx = Context(
            page_size=1,
            attn_backend=self.attn_backend,
            mamba_pool=self.mamba_pool,
            gdn_backend=self.gdn_backend,
        )
        set_global_ctx(self.ctx)
        self.sampler = Sampler(self.model_meta.vocab_size)

        # Spec-decoding engines (EAGLE / DFlash) set this in their own
        # ``__init__``; the base engine never has a draft.
        self.draft_model = None

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

    #: Extra mamba state slots per running req, on top of the base 2
    #: (main slot + radix-cache buffer).  Spec engines bump this to 1:
    #: target verify replays the ``K+1`` window into one scratch slot
    #: and the accepted prefix is replayed back into the main slot
    #: once the accept counts are known (no per-token snapshots).
    extra_mamba_slots_per_req: int = 0

    def _verify_width(self, config: EngineConfig) -> int:
        """Spec-decoding hook: verify window width (W = K + 1) per req.

        The mamba pool allocates per-layer conv window buffers of
        width ``W`` when this is non-zero; spec engines override it.
        """
        return 0

    def _create_mamba_pool(self, config: EngineConfig):
        from mlx_serve.kvcache.mamba_pool import MambaStateConfig, MambaStatePool

        conv_shapes, temporal_shapes = self.model.get_linear_state_shapes()
        num_linear_layers = len(conv_shapes)
        # Default heuristic: 2x running reqs is enough for normal
        # serving (main slot + radix cache buffer); spec engines bump
        # this by 1 (the scratch slot used by replay-style target
        # verify).  When ``config.num_mamba_slots`` is set we honour
        # it verbatim — the project is meant to be a teaching
        # codebase, so let the user own this knob if they want to
        # experiment.
        if config.num_mamba_slots is not None:
            num_slots = config.num_mamba_slots
            logger.info(
                "Using explicit num_mamba_slots=%d (override)", num_slots,
            )
        else:
            multiplier = 2 + self.extra_mamba_slots_per_req
            num_slots = config.max_running_req * multiplier
            logger.info(
                "Auto-sizing mamba pool: num_slots=%d "
                "(= max_running_req * (2 + %d extra))",
                num_slots,
                self.extra_mamba_slots_per_req,
            )
        pool_config = MambaStateConfig(
            num_slots=num_slots,
            num_layers=num_linear_layers,
            conv_shapes=conv_shapes,
            temporal_shapes=temporal_shapes,
            verify_width=self._verify_width(config),
        )
        return MambaStatePool(pool_config)

    def _determine_num_pages(self, config: EngineConfig) -> int:
        bytes_per_page = (
            2  # key + value
            * self.model_meta.head_dim
            * self.model_meta.num_kv_heads
            * config.page_size
            * 2  # sizeof(float16) == sizeof(bfloat16)
            * self.model_meta.num_layers
        )

        if config.kv_cache_gb is not None:
            kv_bytes = config.kv_cache_gb * (1024 ** 3)
            num_pages = int(kv_bytes / bytes_per_page)
        else:
            max_seq_len = (
                config.max_seq_len_override
                if config.max_seq_len_override is not None
                else self.model_meta.max_position
            )
            num_pages = min(max_seq_len * max(config.max_running_req, 1), 262_144)

        assert num_pages > 1, (
            "Not enough memory for KV cache. "
            "Increase --kv-cache-gb or reduce model size."
        )
        real_kv_size = num_pages * bytes_per_page / (1024 ** 3)
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
        return ForwardOutput(next_tokens=next_tokens.astype(mx.int32))

    def shutdown(self) -> None:
        pass