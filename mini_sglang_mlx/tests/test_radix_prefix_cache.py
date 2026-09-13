"""Radix prefix cache: topology, KV page accounting and mamba boundaries.

The interesting part of a hybrid model is that KV is a *sequence* of pages
while the recurrent state is a single indivisible blob that can only ever
be attached to a node boundary.  These tests pin down where the blob may
live and what a match can reuse.
"""

from __future__ import annotations

import pytest

from mini_sglang_mlx.tests._factories import (
    cache_prefix,
    ids,
    make_hybrid,
    make_radix,
    node_of,
    page_balance,
    pending,
    pool_balance,
    set_lru_order,
)

PREFIX = [1, 2, 3, 4, 5, 6]


def test_match_prefix_is_a_pure_query():
    manager, pool = make_hybrid(num_slots=4)
    slot = pool.alloc()
    cache_prefix(manager, PREFIX, mamba_slot=slot)

    before = (pool.available_size, manager.manager.size_info, manager.available_size)
    for _ in range(3):
        handle, indices = manager.manager.match_prefix(ids(*PREFIX))
        assert handle.cached_len == len(PREFIX)
        assert len(indices) == len(PREFIX)
        # A query that forked a slot would drain the pool one call at a time.
        assert (
            pool.available_size,
            manager.manager.size_info,
            manager.available_size,
        ) == before


def test_a_hit_reports_its_snapshot_and_pages():
    manager, pool = make_hybrid(num_slots=4)
    slot = pool.alloc()
    pages = cache_prefix(manager, PREFIX, mamba_slot=slot)

    handle, indices = manager.match_req(pending(0, PREFIX + [7]))
    assert handle.cached_len == len(PREFIX)
    assert handle.node.mamba_slot == slot
    assert indices.tolist() == pages.tolist()


def test_split_leaves_the_snapshot_on_the_deeper_node():
    manager, pool = make_hybrid(num_slots=4)
    slot_a, slot_b = pool.alloc(), pool.alloc()
    cache_prefix(manager, PREFIX, mamba_slot=slot_a)
    cache_prefix(manager, [1, 2, 3, 9], mamba_slot=slot_b)

    # The split point became a tombstone: state is not splittable.
    assert node_of(manager, [1, 2, 3]).mamba_slot is None
    assert node_of(manager, [1, 2, 3, 4, 5, 6]).mamba_slot == slot_a
    assert node_of(manager, [1, 2, 3, 9]).mamba_slot == slot_b
    assert sorted(manager.manager.collect_mamba_slots()) == sorted([slot_a, slot_b])

    # A request ending exactly at the tombstone cannot use the state of a
    # longer prefix, so it gets no reuse at all -- even though its KV is
    # cached, the state at that boundary does not exist.
    handle, _ = manager.manager.match_prefix(ids(1, 2, 3))
    assert handle.cached_len == 0

    # Deeper, and the whole prefix comes back.
    handle, indices = manager.manager.match_prefix(ids(*PREFIX))
    assert handle.cached_len == len(PREFIX)
    assert handle.node.mamba_slot == slot_a
    assert len(indices) == len(PREFIX)


def test_locking_moves_the_prefix_out_of_the_evictable_size():
    manager, pool = make_hybrid(num_slots=4)
    cache_prefix(manager, PREFIX, mamba_slot=pool.alloc())
    handle, _ = manager.manager.match_prefix(ids(*PREFIX))

    before = manager.available_size
    manager.lock(handle)
    # This is why a budget check has to be repeated after locking.
    assert manager.available_size == before - len(PREFIX)
    assert manager.manager.size_info.protected_size == len(PREFIX)

    manager.unlock(handle)
    assert manager.available_size == before
    page_balance(manager)


def test_eviction_frees_the_snapshot_and_returns_its_pages():
    manager, pool = make_hybrid(num_slots=4)
    slot = pool.alloc()
    cache_prefix(manager, PREFIX, mamba_slot=slot)
    free_before, held_before = page_balance(manager)
    assert held_before == len(PREFIX)

    # Ask for more pages than are free: the only evictable leaf goes.
    handed_out = manager.allocate(free_before + 2)
    assert pool.available_size == 4
    assert manager.manager.collect_mamba_slots() == []
    assert manager.manager.root_node.is_leaf()

    manager._free(handed_out)
    page_balance(manager)


def test_evict_for_mamba_reclaims_in_lru_order():
    manager, pool = make_hybrid(num_slots=4)
    for path in ([1, 2], [10, 11], [20, 21]):
        cache_prefix(manager, path, mamba_slot=pool.alloc())
    assert pool.available_size == 1
    set_lru_order(manager, [[1, 2], [10, 11], [20, 21]])

    assert manager.evict_for_mamba(1) == 1
    assert pool.available_size == 2
    with pytest.raises(KeyError):
        node_of(manager, [1, 2])
    assert node_of(manager, [10, 11]).mamba_slot is not None
    page_balance(manager)

    assert manager.evict_for_mamba(1) == 1
    with pytest.raises(KeyError):
        node_of(manager, [10, 11])
    page_balance(manager)


def test_evict_for_mamba_walks_through_slot_less_leaves():
    manager, pool = make_hybrid(num_slots=4)
    cache_prefix(manager, PREFIX, mamba_slot=pool.alloc())
    cache_prefix(manager, [1, 2, 3, 9], mamba_slot=pool.alloc())
    # Evict the tombstone's children (3 + 1 pages), leaving it a leaf that
    # owns no slot.  Evicting the full 7 would take the tombstone too.
    manager._free(manager.manager.evict(4))
    tombstone = node_of(manager, [1, 2, 3])
    assert tombstone.mamba_slot is None and tombstone.is_leaf()
    assert pool.available_size == 4

    cache_prefix(manager, [9], mamba_slot=pool.alloc())
    set_lru_order(manager, [[1, 2, 3]])  # colder than the [9] leaf
    _, held_before = page_balance(manager)

    # The coldest leaf has no snapshot, so the walk evicts it first.  That
    # KV loss buys nothing by itself, but it is the LRU order -- preferring
    # a hotter leaf just because it owns a slot would be wrong.
    assert manager.evict_for_mamba(1) == 1
    _, held_after = page_balance(manager)
    assert held_before - held_after == 4  # 3 tombstone pages + 1 leaf page
    assert manager.manager.collect_mamba_slots() == []


def test_evict_for_mamba_drains_gracefully():
    manager, pool = make_hybrid(num_slots=4)
    assert manager.evict_for_mamba(2) == 0
    page_balance(manager)

    cache_prefix(manager, [1, 2], mamba_slot=pool.alloc())
    assert manager.evict_for_mamba(5) == 1  # more than exists
    assert manager.evict_for_mamba(1) == 0
    page_balance(manager)


def test_coldest_snapshot_node_skips_only_the_node_itself():
    manager, pool = make_hybrid(num_slots=4)
    slots = [pool.alloc() for _ in range(3)]
    cache_prefix(manager, [1, 2], mamba_slot=slots[0])
    cache_prefix(manager, [1, 2, 3, 4], mamba_slot=slots[1])
    cache_prefix(manager, [1, 2, 3, 4, 5, 6], mamba_slot=slots[2])

    coldest = manager.manager.coldest_snapshot_node()
    assert coldest is node_of(manager, [1, 2])
    # Excluding the coldest must not hide its descendants from the search.
    assert manager.manager.coldest_snapshot_node(exclude=coldest) is node_of(
        manager, [1, 2, 3, 4]
    )
    assert (
        manager.manager.coldest_snapshot_node(
            exclude=node_of(manager, [1, 2, 3, 4, 5, 6])
        )
        is coldest
    )


def test_check_integrity_detects_a_slot_that_escaped_the_tree():
    manager, pool = make_hybrid(num_slots=3)
    cache_prefix(manager, PREFIX, mamba_slot=pool.alloc())
    manager.check_integrity()

    leaked = pool.alloc()  # nothing records it: a leak by construction
    with pytest.raises(RuntimeError, match="leaked"):
        manager.check_integrity()
    pool.free(leaked)
    manager.check_integrity()
    pool_balance(manager)


def test_plain_radix_manager_keeps_the_evict_contract():
    manager = make_radix(num_pages=32)
    pages = manager.allocate(6)
    assert manager.manager.insert_prefix(ids(*PREFIX), pages) == 0
    assert manager.manager.evict(0).shape == (0,)
    assert manager.manager._release_node_state(
        manager.manager.root_node
    ) == 0  # no per-node state without a pool

    with pytest.raises(AssertionError, match="Cannot evict"):
        manager.manager.evict(7)
    manager._free(manager.manager.evict(6))
    page_balance(manager)


def test_insert_prefix_refuses_to_park_a_snapshot_on_the_root():
    manager, pool = make_hybrid(num_slots=2)
    empty = ids()
    with pytest.raises(AssertionError, match="snapshot on the root"):
        manager.manager.insert_prefix(empty, empty, mamba_slot=pool.alloc())
