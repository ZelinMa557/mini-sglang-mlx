from __future__ import annotations

import mlx.core as mx

from mlx_serve_kernel import store_kv_cache

from .base import BaseKVCache, KVCacheLayout


class MHAKVCache(BaseKVCache):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: mx.Dtype,
        kv_layout: KVCacheLayout,
        device: None = None,
    ):
        del device  # MLX runs on Apple Silicon, no explicit device
        match kv_layout:
            case KVCacheLayout.PageFirst:
                kv_buffer = mx.empty(
                    (2, num_pages, num_layers, num_kv_heads, head_dim),
                    dtype=dtype,
                )
                kv_buffer = mx.transpose(kv_buffer, (0, 2, 1, 3, 4))
            case KVCacheLayout.LayerFirst:
                kv_buffer = mx.empty(
                    (2, num_layers, num_pages, num_kv_heads, head_dim),
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        self._kv_buffer = mx.reshape(
            kv_buffer, (2, num_layers, num_pages, 1, num_kv_heads, head_dim)
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages, num_kv_heads, head_dim)

    def k_cache(self, index: int) -> mx.array:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> mx.array:
        return self._v_buffer[index]

    def store_kv(
        self, k: mx.array, v: mx.array, out_loc: mx.array, layer_id: int
    ) -> None:
        k_cache = self._k_buffer[layer_id]
        v_cache = self._v_buffer[layer_id]
        mx.eval(k, v, out_loc)
        store_kv_cache(k_cache=k_cache, v_cache=v_cache, indices=out_loc, k=k, v=v)

    @property
    def dtype(self) -> mx.Dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
