from __future__ import annotations

from typing import Tuple

import mlx.core as mx

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    pass


class NaiveCacheManager(BaseCacheManager):
    def __init__(self, device: None = None):
        del device  # MLX runs on Apple Silicon, no explicit device
        self.empty_tensor = mx.array([], dtype=mx.int32)
        super().__init__()

    def match_prefix(self, input_ids: mx.array) -> Tuple[NaiveCacheHandle, mx.array]:
        _ = input_ids  # unused
        return NaiveCacheHandle(0), self.empty_tensor

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        _ = handle, unlock  # unused

    def insert_prefix(self, input_ids: mx.array, indices: mx.array) -> int:
        assert len(indices) == len(input_ids)
        return len(indices)

    def evict(self, size: int) -> mx.array:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
