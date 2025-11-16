"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import numpy as np
import mlx.core as mx


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
    ):

        self.size = size
        self.max_context_len = max_context_len
        self.req_to_token = mx.zeros(
            (size, max_context_len), dtype=mx.int32
        )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: mx.Dtype,
        layer_num: int,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = dtype
        self.layer_num = layer_num

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> mx.array:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> mx.array:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[mx.array, mx.array]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer_id: int,
        loc: mx.array,
        cache_k: mx.array,
        cache_v: mx.array,
    ) -> None:
        raise NotImplementedError()

    def get_flat_data(self, indices):
        raise NotImplementedError()

    def transfer(self, indices, flat_data):
        raise NotImplementedError()

    def transfer_per_layer(self, indices, flat_data, layer_id):
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter


class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: mx.Dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
        )

        self.head_num = head_num
        self.head_dim = head_dim

        self._create_buffers()

        self.layer_transfer_counter = None

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            mx.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            mx.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
            )
            for _ in range(self.layer_num)
        ]

        self.data_strides = mx.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: mx.array,
        cache_k: mx.array,
        cache_v: mx.array,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        self.k_buffer[layer_id][loc] = cache_k
        self.v_buffer[layer_id][loc] = cache_v