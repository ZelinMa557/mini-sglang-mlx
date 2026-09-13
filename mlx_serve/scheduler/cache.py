from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_serve.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from mlx_serve.kvcache.mamba_pool import MambaStatePool
    from mlx_serve.kvcache.radix_manager import (
        HybridCacheHandle,
        HybridRadixCacheManager,
        RadixTreeNode,
    )

    from .utils import PendingReq


class CacheManager:
    def __init__(self, device: None, num_pages: int, type: str):
        self._free_slots = mx.arange(num_pages, dtype=mx.int32)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: mx.array) -> None:
        if len(indices) > 0:
            self._free_slots = mx.concatenate([self._free_slots, indices])

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> mx.array:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        evicted = self.manager.evict(needed_len - free_len)
        merged = mx.concatenate([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: mx.array,
        indices: mx.array,
        mamba_slot: int | None = None,
    ) -> None:
        if mamba_slot is not None:
            from mlx_serve.kvcache.radix_manager import HybridRadixCacheManager

            assert isinstance(self.manager, HybridRadixCacheManager)
            in_cache_len = self.manager.insert_prefix(
                input_ids, indices, mamba_slot=mamba_slot
            )
        else:
            in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )


class HybridCacheManager(CacheManager):
    """CacheManager that also manages MambaStatePool for hybrid models."""

    def __init__(
        self,
        device: None,
        num_pages: int,
        mamba_pool: "MambaStatePool",
    ):
        self._free_slots = mx.arange(num_pages, dtype=mx.int32)
        self.device = device
        self.num_pages = num_pages
        self.mamba_pool = mamba_pool

        from mlx_serve.kvcache import create_hybrid_radix_cache_manager

        self.manager = create_hybrid_radix_cache_manager(
            mamba_pool=mamba_pool, device=device
        )

    def free_mamba_slot(self, slot: int | None) -> None:
        """Return a directly-held slot to the pool.

        Used when a slot is given up without the request that held it
        running to completion (see :meth:`_shed_cached_snapshot`).  A
        request that *finishes* hands its slot to the radix tree instead
        (``free_and_cache_finished_req``), which is why this is not on that
        path.
        """
        if slot is not None:
            self.mamba_pool.free(slot)

    def acquire_mamba_slot(self, handle: "HybridCacheHandle") -> int | None:
        """Hand out a pool slot for a request admitted on *handle*.

        On a prefix hit the snapshot is *copied* into the new slot, so the
        node keeps its own state and both requests can run independently.

        Must be called with *handle* already locked: this may evict, and an
        unlocked node could lose the very snapshot being copied.  Returns
        ``None`` only when the pool is dry and no cached snapshot can be
        given up, in which case the caller waits for another round.
        """
        slot = self.mamba_pool.alloc()
        if slot is None:
            # A dry pool is cache pressure like any other: reclaim snapshots
            # from the coldest prefixes instead of failing.
            self.evict_for_mamba(1)
            slot = self.mamba_pool.alloc()
        if slot is None:
            # Nothing is evictable, which means every remaining snapshot
            # sits on a chain locked by a live request.  State reuse is
            # best-effort, so give one up rather than stall: the request
            # that locked it forked its own copy at admission and will not
            # read the snapshot again.
            slot = self._shed_cached_snapshot(keep=handle.node)
        if slot is None:
            return None

        if handle.cached_len > 0:
            src = handle.node.mamba_slot
            assert src is not None, "handle claims a cached prefix with no state"
            self.mamba_pool.copy(src, slot)
        return slot

    def _shed_cached_snapshot(self, keep: "RadixTreeNode") -> int | None:
        """Give up the coldest cached snapshot (except *keep*'s) for reuse.

        The node keeps its KV pages and only loses its recurrent state, so
        a later request matching it simply falls back to the nearest
        ancestor that still has a snapshot -- a shallower reuse instead of
        no reuse.  ``keep`` is the node the caller is about to fork from,
        which must survive until the copy is done.
        """
        node = self.manager.coldest_snapshot_node(exclude=keep)
        if node is None:
            return None
        self.free_mamba_slot(node.mamba_slot)
        node.mamba_slot = None
        return self.mamba_pool.alloc()

    def evict_for_mamba(self, count: int) -> int:
        """Free at least *count* mamba slots by evicting cached prefixes.

        Returns the number of slots actually reclaimed, which falls short
        of *count* once there is nothing evictable left.
        """
        from mlx_serve.kvcache.radix_manager import HybridRadixCacheManager

        assert isinstance(self.manager, HybridRadixCacheManager)
        indices, freed = self.manager.evict_for_mamba(count)
        self._free(indices)
        return freed
