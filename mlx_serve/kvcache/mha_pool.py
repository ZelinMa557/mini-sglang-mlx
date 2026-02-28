from __future__ import annotations

import mlx.core as mx

from mlx_serve_kernel import store_kv_cache

from .base import BaseKVCache


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
    ):
        self._num_layers = num_layers
        self._k_buffer = [mx.zeros((num_pages, num_kv_heads, head_dim), dtype=dtype) for _ in range(num_layers)]
        self._v_buffer = [mx.zeros((num_pages, num_kv_heads, head_dim), dtype=dtype) for _ in range(num_layers)]
        self._storage_shape = (num_pages, num_kv_heads, head_dim)

    def k_cache(self, index: int) -> mx.array:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> mx.array:
        return self._v_buffer[index]

    def store_kv(
        self, k: mx.array, v: mx.array, out_loc: mx.array, layer_id: int
    ) -> None:
        assert out_loc.dtype == mx.int32, "out_loc must be int32"
        k_cache = self._k_buffer[layer_id]
        v_cache = self._v_buffer[layer_id]

        if len(out_loc) != k.shape[0] or len(out_loc) != v.shape[0]:
            raise ValueError(
                f"store_kv shape mismatch: len(out_loc)={len(out_loc)}, "
                f"k.shape={k.shape}, v.shape={v.shape}"
            )
        mx.eval(k, v, out_loc, k_cache, v_cache)
        store_kv_cache(k_cache=k_cache, v_cache=v_cache, indices=out_loc, k=k, v=v)

    @property
    def dtype(self) -> mx.Dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
