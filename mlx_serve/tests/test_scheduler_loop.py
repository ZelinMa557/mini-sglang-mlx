"""The whole scheduler state machine, driven for far longer than the pool is deep.

:class:`Harness` mirrors ``Scheduler._non_spec_loop_step`` --
:meth:`~mlx_serve.scheduler.prefill.PrefillManager.schedule_next_batch`,
``_prepare_batch``, ``Engine.forward_batch`` (stubbed), ``_process_last_data``
-- against the real managers.  The stub engine is what makes the model-free
loop interesting: instead of a forward pass it asserts that the recurrent
state in each request's slot is exactly what a fresh prefill of
``input_ids[:cached_len]`` would have produced.  That one invariant covers
prefix forking, chunked-prefill continuity and slot reuse at once, because
it is false the moment any of them hands a request the wrong state.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_serve.core import Batch
from mlx_serve.scheduler.decode import DecodeManager
from mlx_serve.scheduler.prefill import ChunkedReq, PrefillManager
from mlx_serve.tests._factories import (
    make_hybrid,
    make_table,
    page_balance,
    pending,
    pool_balance,
)

MAX_STEPS = 20_000


def token_sum(tokens) -> float:
    """The stub's notion of "recurrent state after reading *tokens*"."""
    return float(sum(tokens))


def next_token(uid: int, device_len: int) -> int:
    return (uid * 31 + device_len * 7) % 1000


class Harness:
    """The scheduler's per-iteration work, minus the model and the sockets."""

    def __init__(
        self,
        *,
        num_pages: int = 256,
        num_slots: int = 4,
        max_running_reqs: int = 3,
        prefill_budget: int = 16,
    ) -> None:
        self.manager, self.pool = make_hybrid(
            num_pages=num_pages, num_slots=num_slots
        )
        self.table = make_table(max_running_reqs=max_running_reqs, max_seq_len=256)
        self.page_table = self.table.page_table
        self.decode = DecodeManager()
        self.prefill = PrefillManager(self.manager, self.table, self.decode)
        self.prefill_budget = prefill_budget

        self.finished = 0
        self.steps = 0
        self.starved_steps = 0
        self.prefix_hits = 0
        self.evict_calls = 0
        self.slots_handed_out = 0
        self.sequences: dict[int, list[int]] = {}
        self._tokens: dict[int, int] = {}

        inner_match = self.manager.match_req

        def counting_match(req):
            handle, indices = inner_match(req)
            self.prefix_hits += handle.cached_len > 0
            return handle, indices

        inner_acquire = self.manager.acquire_mamba_slot

        def counting_acquire(handle):
            slot = inner_acquire(handle)
            self.slots_handed_out += slot is not None
            return slot

        inner_evict = self.manager.evict_for_mamba

        def counting_evict(count):
            self.evict_calls += 1
            return inner_evict(count)

        self.manager.match_req = counting_match
        self.manager.acquire_mamba_slot = counting_acquire
        self.manager.evict_for_mamba = counting_evict

    # ── inspection ──────────────────────────────────────────────────────

    def held_reqs(self) -> list:
        """Requests that own a mamba slot right now (decoding or mid-chunk)."""
        reqs = list(self.decode.running_reqs)
        reqs += [p.chunked_req for p in self.prefill.pending_list if p.chunked_req]
        return reqs

    def held_pages(self) -> int:
        """KV pages a live request allocated for itself.

        Everything below ``cache_handle.cached_len`` came from the tree and
        is still the tree's; the rest of the row is this request's own.
        """
        return sum(
            req.cached_len - req.cache_handle.cached_len for req in self.held_reqs()
        )

    # ── driving ─────────────────────────────────────────────────────────

    def submit(self, uid: int, tokens, max_tokens: int = 3) -> None:
        self.prefill.add_one_req(pending(uid, tokens, max_tokens))

    def _set_state(self, slot: int, tokens) -> None:
        self.pool.temporal_state(0)[slot] = token_sum(tokens)

    def _forward(self, batch: Batch) -> None:
        """What ``Engine.forward_batch`` plus the GDN kernel do, as one assert."""
        for req in batch.reqs:
            assert req.mamba_slot is not None
            state = self.pool.temporal_state(0)[req.mamba_slot].min().item()
            expected = token_sum(req.input_ids[: req.cached_len].tolist())
            assert state == expected, (
                f"uid={req.uid} started with state {state} but its cached "
                f"prefix of {req.cached_len} tokens should give {expected}"
            )
            committed = req.device_len
            req.complete_one()  # Engine.forward_batch does this for every req
            self._set_state(req.mamba_slot, req.input_ids[:committed].tolist())
            self._tokens[req.uid] = next_token(req.uid, committed)

    def _prepare_batch(self, batch: Batch) -> None:
        needed = sum(req.extend_len for req in batch.reqs)
        batch.out_loc = self.manager.allocate(needed)
        offset = 0
        for req in batch.reqs:
            if req.extend_len > 0:
                self.page_table[req.table_idx, req.cached_len : req.device_len] = (
                    batch.out_loc[offset : offset + req.extend_len]
                )
                offset += req.extend_len

    def _process_last_data(self, batch: Batch) -> None:
        finished = []
        for req in batch.reqs:
            if isinstance(req, ChunkedReq):
                continue  # the scheduler does not sample a chunk
            req.append_host(mx.array([self._tokens[req.uid]], dtype=mx.int32))
            if not req.can_decode():
                finished.append(req)
                self.decode.remove_req(req)

        for req in finished:
            self.table.free(req.table_idx)
            self.manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
                mamba_slot=req.mamba_slot,
            )
            self.sequences[req.uid] = req.input_ids[: req.cached_len].tolist()
        self.finished += len(finished)

    def step(self) -> bool:
        if self.prefill.runnable and self.pool.available_size == 0:
            self.starved_steps += 1  # a pending req is waiting on a slot

        batch = self.prefill.schedule_next_batch(
            self.prefill_budget
        ) or self.decode.schedule_next_batch()
        if batch is None:
            return False
        self.steps += 1
        self._prepare_batch(batch)
        self._forward(batch)
        self.decode.filter_reqs(batch.reqs)
        self._process_last_data(batch)
        return True

    def drain(self) -> None:
        while self.prefill.runnable or self.decode.runnable:
            assert self.steps < MAX_STEPS, (
                "the scheduler stopped making progress with "
                f"{len(self.prefill.pending_list)} reqs pending and "
                f"{self.pool.available_size} mamba slots free"
            )
            assert self.step(), "no batch could be built, but work remains"
        self.manager.check_integrity()


def assert_balanced(harness: Harness) -> None:
    """Every slot has exactly one owner, and every page is free xor held."""
    free = set(harness.pool._free_slots)
    tree = set(harness.manager.manager.collect_mamba_slots())
    live = {req.mamba_slot for req in harness.held_reqs()} - {None}
    assert not (free & tree), "a snapshot is both freed and cached"
    assert not (free & live), "a live request holds a freed slot"
    assert not (tree & live), "a live request's slot is also cached"
    assert free | tree | live == set(range(1, harness.pool.num_slots + 1)), (
        "unowned mamba slots: "
        f"{sorted(set(range(1, harness.pool.num_slots + 1)) - (free | tree | live))}"
    )

    free_pages = len(harness.manager._free_slots)
    tree_pages = harness.manager.manager.size_info.total_size
    live_pages = harness.held_pages()
    assert free_pages + tree_pages + live_pages == harness.manager.num_pages, (
        f"page accounting broken: {free_pages} free + {tree_pages} cached + "
        f"{live_pages} in flight != {harness.manager.num_pages} pages"
    )


def test_many_requests_through_a_small_pool():
    harness = Harness(num_slots=4, max_running_reqs=3, prefill_budget=16)
    # Prompts longer than the budget, so every one of them is chunked.
    for uid in range(20):
        harness.submit(uid, [uid * 100] + list(range(19)))
    harness.drain()

    assert harness.finished == 20
    assert harness.slots_handed_out == 20  # one per request, never more
    assert harness.evict_calls > 0  # the pool ran dry and reclaimed prefixes
    assert_balanced(harness)
    # Nothing here can share a cached boundary: every prompt starts with a
    # different token, so this run is all cold prefill.
    assert harness.prefix_hits == 0


def test_a_replayed_sequence_forks_the_cached_state_instead_of_recomputing_it():
    harness = Harness(num_slots=4, max_running_reqs=2, prefill_budget=16)
    harness.submit(0, [1] + list(range(19)), max_tokens=4)
    harness.drain()
    cached = harness.sequences[0]
    assert len(cached) > 20  # the prompt plus what it generated

    # Continuing that exact sequence lands the match on the stored boundary,
    # so the state there is forked rather than recomputed -- and the assert
    # in ``_forward`` checks the fork really carries that state.
    harness.submit(1, cached + [999], max_tokens=2)
    harness.drain()

    assert harness.prefix_hits == 1
    assert harness.finished == 2
    assert harness.slots_handed_out == 2
    assert_balanced(harness)


def test_a_long_prefill_chunks_and_keeps_its_recurrent_state_across_chunks():
    harness = Harness(num_slots=2, max_running_reqs=1, prefill_budget=4)
    harness.submit(0, list(range(30)), max_tokens=2)
    harness.drain()

    # 30 tokens at 4 per chunk: 7 chunked forwards before the last one
    # completes the prompt.  A per-chunk slot (the old behaviour) would
    # restart the recurrence from zero and trip ``_forward``'s assert.
    assert harness.steps > 8
    assert harness.slots_handed_out == 1
    assert harness.finished == 1
    assert_balanced(harness)


def test_slot_ownership_stays_exclusive_across_a_long_run():
    harness = Harness(num_slots=3, max_running_reqs=3, prefill_budget=8)
    for uid in range(12):
        harness.submit(uid, [uid * 50] + list(range(11)))

    while harness.prefill.runnable or harness.decode.runnable:
        harness.step()
        assert_balanced(harness)

    harness.manager.check_integrity()
    assert harness.finished == 12
    assert harness.slots_handed_out == 12  # 12 requests through 3 slots


def test_a_dry_pool_with_nothing_to_reclaim_waits_for_a_slot_to_come_back():
    """No deadlock: a blocked admission must not stop the running request."""
    harness = Harness(num_slots=1, max_running_reqs=2, prefill_budget=64)
    harness.submit(0, list(range(6)), max_tokens=2)
    harness.submit(1, list(range(100, 106)), max_tokens=2)
    harness.drain()

    assert harness.finished == 2  # the second waited for the first
    assert harness.starved_steps > 0
    assert harness.slots_handed_out == 2
    assert_balanced(harness)
