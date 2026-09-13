"""Admission control for hybrid models: who gets a mamba slot, and giving it back.

The pool used to drain one slot per rejected admission, one per prefill
chunk, and never came back from a full cache.  Every test here is a
regression on one of those, driven through the real
:class:`PrefillAdder` / :class:`PrefillManager`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mini_sglang_mlx.scheduler.prefill import ChunkedReq, PrefillAdder, PrefillManager
from mini_sglang_mlx.tests._factories import (
    cache_prefix,
    make_hybrid,
    make_table,
    node_of,
    page_balance,
    pending,
    pool_balance,
    set_lru_order,
)


@dataclass
class _IdleDecode:
    """Stand-in for :class:`DecodeManager` with nothing decoding."""

    inflight_tokens: int = 0
    runnable: bool = False


def make_adder(manager, table, *, reserved_size: int, token_budget: int = 100):
    return PrefillAdder(
        token_budget=token_budget,
        reserved_size=reserved_size,
        cache_manager=manager,
        table_manager=table,
    )


def test_a_rejected_admission_gives_back_everything_it_took():
    """The bug that made the pool drain one snapshot per retry."""
    manager, pool = make_hybrid(num_pages=64, num_slots=4)
    table = make_table(max_running_reqs=2)
    prefix = list(range(10))
    cache_prefix(manager, prefix, mamba_slot=pool.alloc())

    req = pending(0, prefix + [7, 8], max_tokens=3)
    # Locking moves the 10 matched tokens out of the evictable size, so this
    # budget passes before the lock (5 + 55 <= 64) and fails after it
    # (60 > 54).  The rejection is exactly the interesting one.
    adder = make_adder(manager, table, reserved_size=55)

    for _ in range(5):  # the scheduler retries the same req every iteration
        assert adder.try_add_one(req) is None
        assert pool.available_size == 3
        assert manager.available_size == 64
        assert table.available_size == 2
    page_balance(manager)
    pool_balance(manager)


def test_chunked_prefill_keeps_one_slot_and_its_state_across_chunks():
    """Each chunk used to allocate a fresh, zeroed slot mid-prefill."""
    manager, pool = make_hybrid(num_pages=64, num_slots=4)
    table = make_table(max_running_reqs=2)
    prefill = PrefillManager(manager, table, _IdleDecode())

    prefill.add_one_req(pending(0, range(20), max_tokens=2))
    first = prefill.schedule_next_batch(prefill_budget=8)
    assert first is not None and len(first.reqs) == 1
    chunk = first.reqs[0]
    assert isinstance(chunk, ChunkedReq)
    assert pool.available_size == 3
    slot = chunk.mamba_slot
    assert slot is not None

    # Whatever the engine accumulated into the slot during chunk 1 must
    # survive into chunk 2 -- the recurrent state is the whole point.
    pool.temporal_state(0)[slot] = 1.0
    chunk.complete_one()  # the engine commits the chunk it just ran

    second = prefill.schedule_next_batch(prefill_budget=8)
    assert second is not None
    following = second.reqs[0]
    assert following.mamba_slot == slot
    assert following.table_idx == chunk.table_idx
    assert pool.available_size == 3  # no second slot was taken
    assert pool.temporal_state(0)[slot].min().item() == 1.0  # not re-zeroed


def test_admission_waits_instead_of_asserting_when_the_pool_is_pinned():
    manager, pool = make_hybrid(num_pages=64, num_slots=2)
    table = make_table(max_running_reqs=3)
    adder = make_adder(manager, table, reserved_size=0)

    live = [
        adder.try_add_one(pending(uid, range(uid * 10, uid * 10 + 4)))
        for uid in (1, 2)
    ]
    assert all(req is not None for req in live)
    assert pool.available_size == 0

    # Nothing cached to evict and no free slot: the request must wait for
    # another round rather than take the scheduler process down.
    assert adder.try_add_one(pending(3, range(90, 94))) is None
    assert pool.available_size == 0
    assert table.available_size == 1  # the table row was not consumed either
    pool_balance(manager, live_slots=[req.mamba_slot for req in live])


def test_kv_pressure_evicts_a_cached_prefix_to_free_a_slot():
    manager, pool = make_hybrid(num_pages=64, num_slots=2)
    table = make_table(max_running_reqs=2)
    cache_prefix(manager, [1, 2, 3, 4], mamba_slot=pool.alloc())
    cache_prefix(manager, [10, 11, 12, 13, 14], mamba_slot=pool.alloc())
    assert pool.available_size == 0
    set_lru_order(manager, [[1, 2, 3, 4], [10, 11, 12, 13, 14]])

    adder = make_adder(manager, table, reserved_size=0)
    req = adder.try_add_one(pending(0, [20, 21, 22], max_tokens=2))

    assert req is not None and req.mamba_slot is not None
    with pytest.raises(KeyError):
        node_of(manager, [1, 2, 3, 4])  # the colder prefix paid for it
    assert node_of(manager, [10, 11, 12, 13, 14]).mamba_slot is not None
    assert manager.manager.collect_mamba_slots() == [node_of(manager, [10, 11, 12, 13, 14]).mamba_slot]
    page_balance(manager)
    pool_balance(manager, live_slots=[req.mamba_slot])


def test_a_pinned_chain_gives_up_a_snapshot_rather_than_stalling():
    """Multi-turn traffic: one live req can lock a whole chain of snapshots.

    With every snapshot on a locked chain nothing is evictable, so without
    giving one up the pool could never refill and admission would spin
    forever.
    """
    manager, pool = make_hybrid(num_pages=64, num_slots=2)
    table = make_table(max_running_reqs=2)
    older_slot = pool.alloc()
    cache_prefix(manager, [1, 2, 3, 4], mamba_slot=older_slot)
    newest_slot = pool.alloc()
    cache_prefix(manager, [1, 2, 3, 4, 5, 6], mamba_slot=newest_slot)
    assert pool.available_size == 0
    pool.temporal_state(0)[newest_slot] = 2.0

    adder = make_adder(manager, table, reserved_size=0)
    req = adder.try_add_one(pending(0, [1, 2, 3, 4, 5, 6, 7], max_tokens=2))

    assert req is not None and req.mamba_slot is not None
    # The prefix it matched keeps its snapshot -- that is the one being
    # forked -- and the older one on the same locked chain is what goes.
    assert node_of(manager, [1, 2, 3, 4, 5, 6]).mamba_slot == newest_slot
    assert node_of(manager, [1, 2, 3, 4]).mamba_slot is None
    # It reused the matched prefix, with a real copy of the state.
    assert req.cached_len == 6
    assert pool.temporal_state(0)[req.mamba_slot].min().item() == 2.0
    page_balance(manager)
    pool_balance(manager, live_slots=[req.mamba_slot])


def test_a_finished_request_hands_its_slot_to_the_cache_and_the_next_one_forks_it():
    manager, pool = make_hybrid(num_pages=64, num_slots=4)
    table = make_table(max_running_reqs=2)
    prefill = PrefillManager(manager, table, _IdleDecode())
    prompt = list(range(8))

    prefill.add_one_req(pending(0, prompt, max_tokens=1))
    batch = prefill.schedule_next_batch(prefill_budget=32)
    assert batch is not None
    first = batch.reqs[0]
    slot = first.mamba_slot
    assert slot is not None

    # What the engine does for a prefill batch, then the finish path.
    table.page_table[first.table_idx, first.cached_len : first.device_len] = (
        manager.allocate(first.extend_len)
    )
    first.complete_one()
    assert not first.can_decode()
    table.free(first.table_idx)
    manager.free_and_cache_finished_req(
        first.cache_handle,
        first.input_ids[: first.cached_len],
        table.page_table[first.table_idx, : first.cached_len],
        mamba_slot=first.mamba_slot,
    )

    # Quiescent: every slot is either free or held by the tree.
    manager.check_integrity()
    pool_balance(manager)
    assert manager.manager.collect_mamba_slots() == [slot]

    pool.temporal_state(0)[slot] = 3.0
    prefill.add_one_req(pending(1, prompt + [9], max_tokens=1))
    batch = prefill.schedule_next_batch(prefill_budget=32)
    assert batch is not None
    second = batch.reqs[0]

    assert second.cached_len == len(prompt)  # the prefix was reused
    assert second.mamba_slot != slot  # forked, not shared
    assert pool.temporal_state(0)[second.mamba_slot].min().item() == 3.0
    assert manager.manager.collect_mamba_slots() == [slot]
    page_balance(manager)
    pool_balance(manager, live_slots=[second.mamba_slot])
