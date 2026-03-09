from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import mlx.core as mx

from mlx_serve.utils import Registry

from .base import (
    BaseCacheHandle,
    BaseCacheManager,
    BaseKVCache,
    SizeInfo,
)

if TYPE_CHECKING:
    from mlx_serve.models import ModelConfig
    from .mamba_pool import MambaStatePool


class CacheManagerCreator(Protocol):
    def __call__(self, device: None = None) -> BaseCacheManager: ...


SUPPORTED_CACHE_MANAGER: Registry[CacheManagerCreator] = Registry("Cache Manager")


def create_kvcache(
    model_config: "ModelConfig",
    num_pages: int,
    dtype: mx.Dtype,
    device: None = None,
) -> BaseKVCache:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


def create_naive_cache_manager(device: None = None) -> BaseCacheManager:
    from .naive_manager import NaiveCacheManager

    return NaiveCacheManager(device=device)


def create_radix_cache_manager(device: None = None) -> BaseCacheManager:
    from .radix_manager import RadixCacheManager

    return RadixCacheManager(device=device)


def create_hybrid_radix_cache_manager(
    mamba_pool: "MambaStatePool", device: None = None
) -> BaseCacheManager:
    from .radix_manager import HybridRadixCacheManager

    return HybridRadixCacheManager(mamba_pool=mamba_pool, device=device)


SUPPORTED_CACHE_MANAGER.register("naive")(create_naive_cache_manager)
SUPPORTED_CACHE_MANAGER.register("radix")(create_radix_cache_manager)


def create_cache_manager(device: None = None, type: str = "naive") -> BaseCacheManager:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache",
    "create_cache_manager",
    "create_hybrid_radix_cache_manager",
    "BaseKVCache",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "SUPPORTED_CACHE_MANAGER",
]
