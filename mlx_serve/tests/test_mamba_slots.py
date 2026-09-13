"""Lifecycle of :class:`MambaStatePool` slots.

A slot is a scarce resource with exactly one owner at a time, so most of
what matters here is that the ownership rules fail loudly instead of
silently letting two sequences share one recurrent state.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_serve.tests._factories import CONV_SHAPE, TEMPORAL_SHAPE, make_pool


def test_alloc_is_lifo_and_never_hands_out_slot_zero():
    pool = make_pool(num_slots=3)
    assert pool.available_size == 3
    slots = [pool.alloc() for _ in range(3)]
    assert slots == [3, 2, 1]  # LIFO, and slot 0 is the dummy
    assert pool.available_size == 0
    assert pool.alloc() is None


def test_free_makes_the_slot_available_again_most_recently_first():
    pool = make_pool(num_slots=2)
    first, second = pool.alloc(), pool.alloc()
    pool.free(first)
    assert pool.alloc() == first
    pool.free(second)
    assert pool.alloc() == second


def test_alloc_zeroes_the_state_it_hands_out():
    pool = make_pool(num_slots=1)
    slot = pool.alloc()
    pool.temporal_state(0)[slot] = mx.full(TEMPORAL_SHAPE, 5.0)
    pool.conv_state(0, 0)[slot] = mx.full(CONV_SHAPE, 5.0)
    assert pool.temporal_state(0)[slot].min().item() == 5.0

    pool.free(slot)
    assert pool.alloc() == slot
    assert pool.temporal_state(0)[slot].max().item() == 0.0
    assert pool.conv_state(0, 0)[slot].max().item() == 0.0


def test_alloc_many_refuses_without_consuming_anything():
    pool = make_pool(num_slots=2)
    assert pool.alloc_many(3) is None
    assert pool.available_size == 2  # the failed call took nothing

    slots = pool.alloc_many(2)
    assert sorted(slots) == [1, 2]
    assert pool.available_size == 0
    assert pool.alloc_many(0) == []


def test_double_free_is_rejected():
    pool = make_pool(num_slots=2)
    slot = pool.alloc()
    pool.free(slot)
    # Two copies in the free list would hand one state to two sequences.
    with pytest.raises(AssertionError, match="already free"):
        pool.free(slot)

    slots = pool.alloc_many(2)
    pool.free_many(slots)
    assert pool.available_size == 2
    with pytest.raises(AssertionError, match="already free"):
        pool.free_many(slots)


@pytest.mark.parametrize("slot", [0, -1, 3])
def test_free_rejects_slots_outside_the_pool(slot):
    pool = make_pool(num_slots=2)
    with pytest.raises(AssertionError, match="Invalid slot"):
        pool.free(slot)


def test_check_ownership_accepts_a_free_list_and_tree_slots():
    pool = make_pool(num_slots=3)
    in_tree = pool.alloc()
    pool.check_ownership([in_tree])
    pool.free(in_tree)
    pool.check_ownership([])


def test_check_ownership_reports_a_slot_nobody_owns():
    pool = make_pool(num_slots=3)
    leaked = pool.alloc()
    with pytest.raises(RuntimeError) as err:
        pool.check_ownership([])
    assert "leaked" in str(err.value)
    assert f"[{leaked}]" in str(err.value)

    pool.free(leaked)
    pool.check_ownership([])


def test_check_ownership_rejects_one_slot_owned_twice():
    pool = make_pool(num_slots=2)
    slot = pool.alloc()
    pool.free(slot)  # now free *and* claimed by the tree
    with pytest.raises(AssertionError, match="owned twice"):
        pool.check_ownership([slot])


def test_copy_duplicates_state_without_aliasing_the_source():
    pool = make_pool(num_slots=2)
    src, dst = pool.alloc(), pool.alloc()
    pool.temporal_state(1)[src] = mx.full(TEMPORAL_SHAPE, 7.0)
    pool.conv_state(1, 0)[src] = mx.full(CONV_SHAPE, 3.0)

    pool.copy(src, dst)
    assert pool.temporal_state(1)[dst].min().item() == 7.0
    assert pool.conv_state(1, 0)[dst].min().item() == 3.0

    # The fork must be independent: mutating the copy leaves the source be.
    pool.temporal_state(1)[dst] = mx.zeros(TEMPORAL_SHAPE)
    assert pool.temporal_state(1)[src].min().item() == 7.0


def test_conv_window_buffers_follow_verify_width():
    plain = make_pool(num_slots=1, verify_width=0)
    with pytest.raises(AssertionError, match="conv windows not allocated"):
        plain.conv_window(0)

    verifier = make_pool(num_slots=1, verify_width=4)
    window = verifier.conv_window(0)
    assert window.shape == (2, 4, *CONV_SHAPE)  # (slots + dummy, steps, ...)
