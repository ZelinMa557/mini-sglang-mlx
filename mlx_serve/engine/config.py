from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import mlx.core as mx
from mlx_serve.utils import cached_load_hf_config

if TYPE_CHECKING:
    from mlx_serve.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    dtype: mx.Dtype
    max_running_req: int = 16
    attention_backend: str = "auto"
    page_size: int = 1
    kv_cache_gb: float | None = None
    max_seq_len_override: int | None = None

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from mlx_serve.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:23333"
