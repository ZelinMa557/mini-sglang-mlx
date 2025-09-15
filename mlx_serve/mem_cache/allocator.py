from __future__ import annotations

"""
Copyright 2025 SGLang Team
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
Page-aligned memory pool.
"""

import abc
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_serve.mem_cache.memory_pool import KVCache

class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: mx.Dtype,
        device: str,
        kvcache: KVCache,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache

        self.free_pages = None

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return len(self.free_pages) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, free_pages):
        self.free_pages = free_pages

    def backup_state(self):
        return self.free_pages

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: mx.array):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(self, size: int, dtype: mx.Dtype, kvcache: KVCache):
        super().__init__(size, 1, dtype, kvcache)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = mx.arange(
            1, self.size + 1, dtype=mx.int64
        )

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages)

    def alloc(self, need_size: int):
        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: mx.array):
        if len(free_index) == 0:
            return
        self.free_pages = mx.concatenate([self.free_pages, free_index])