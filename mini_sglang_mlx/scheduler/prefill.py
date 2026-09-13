from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import mlx.core as mx
from mini_sglang_mlx.core import Batch, BatchPhase, Req
from mini_sglang_mlx.kvcache.radix_manager import HybridCacheHandle
from mini_sglang_mlx.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from mini_sglang_mlx.kvcache import BaseCacheHandle
    from mini_sglang_mlx.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)

_STARVE_WARN_INTERVAL_NS = 5 * 10**9
_last_starve_warn_ns = 0


def _warn_mamba_starved(available: int, num_slots: int) -> None:
    """Report a dry pool that nothing can replenish, rate-limited.

    This is a capacity problem, not a transient wait, so it should be
    visible -- but the scheduler retries every iteration, and a log line
    per retry would drown everything else.
    """
    global _last_starve_warn_ns
    now = time.monotonic_ns()
    if now - _last_starve_warn_ns < _STARVE_WARN_INTERVAL_NS:
        return
    _last_starve_warn_ns = now
    logger.warning(
        "Mamba pool is dry (%d/%d slots free) and no cached prefix is "
        "evictable: requests are waiting on capacity. Raise --num-mamba-slots.",
        available,
        num_slots,
    )


class ChunkedReq(Req):
    def append_host(self, next_token: mx.array) -> None:
        raise NotImplementedError("ChunkedReq should be sampled")

    def can_decode(self) -> bool:
        return False


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(
        self, req: PendingReq
    ) -> Tuple[BaseCacheHandle, int, int | None] | None:
        if self.table_manager.available_size == 0:
            return None

        # A pure query: it neither locks nor allocates anything, so a
        # rejection below cannot leave state behind.
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        # Reachable even though it just passed: locking moves the matched
        # prefix out of the evictable size and into the protected one.
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        mamba_slot: int | None = None
        if isinstance(handle, HybridCacheHandle):
            from .cache import HybridCacheManager

            assert isinstance(self.cache_manager, HybridCacheManager)
            # Deliberately last, and after both budget checks.  This is the
            # only fallible step left, so a rejection past this point would
            # have a slot to give back; and while it may evict, eviction
            # moves pages from the evictable size to the free list without
            # changing the sum the budget is made of, so it cannot
            # invalidate the check just made.  Must come after the lock:
            # an unlocked node could lose the snapshot copied here.
            mamba_slot = self.cache_manager.acquire_mamba_slot(handle)
            if mamba_slot is None:
                self.cache_manager.unlock(handle)
                _warn_mamba_starved(
                    self.cache_manager.mamba_pool.available_size,
                    self.cache_manager.mamba_pool.num_slots,
                )
                return None

        # Cannot fail: the table was known to have a free row before any of
        # the above, and nothing in between allocates from it.
        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids[:] = req.input_ids[:cached_len]
            page_entry[:] = match_indices

        return handle, table_idx, mamba_slot

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        mamba_slot: int | None,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx][_slice]
        device_ids[:] = pending_req.input_ids[_slice]

        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            mamba_slot=mamba_slot,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            # Continuing a chunked prefill: same table row, same handle and
            # the same mamba slot, so the recurrent state accumulates across
            # chunks instead of restarting from zero at each one.
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
                mamba_slot=chunked_req.mamba_slot,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx, mamba_slot = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
                mamba_slot=mamba_slot,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase=BatchPhase.PREFILL)

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
