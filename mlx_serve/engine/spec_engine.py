"""Base class for speculative-decoding engines.

Both EAGLE/MTP and DFlash share a fair amount of plumbing:

* A draft model + its own :class:`Context` / :class:`AttnBackend`
  (subclass owns the cache-shape specifics; pages are MIRRORED with
  the target so the scheduler's :class:`CacheManager` is the single
  source of truth for allocation).
* The per-iter resource lifecycle: allocate ``B*(K+1)`` shared pages,
  drive the draft + target verify, free the K-j unused page columns
  and roll back mamba checkpoints.
* Target prefill / verify metadata builders.

This module factors all of that out so the spec-method-specific work
in each engine stays small and focused on the draft loop itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import mlx.core as mx

from mlx_serve.core import Batch, BatchPhase, Req
from mlx_serve.utils import init_logger

from .config import EngineConfig
from .engine import Engine

if TYPE_CHECKING:
    from mlx_serve.scheduler.cache import CacheManager

logger = init_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# Output container — common across spec methods.
# ════════════════════════════════════════════════════════════════════


@dataclass
class SpecForwardOutput:
    """Per-iter spec-decoding output handed back to the scheduler.

    ``accepted_tokens`` is a list of length ``B``; entry ``i`` is the
    1-D int32 array of NEW tokens committed for req ``i`` this iter.

    * Prefill: ``[T_1]`` (length 1) for every spec method.
    * Decode:  ``[d_1, ..., d_{j}, bonus]`` (length ``j+1``) where
      ``j`` is the number of drafts accepted.  These are the tokens
      NOT already present in ``req.input_ids`` before the iter —
      they are appended to ``req.input_ids`` by the engine before
      returning.
    """

    accepted_tokens: List[mx.array]


# ════════════════════════════════════════════════════════════════════
# SpecEngine — abstract base.
# ════════════════════════════════════════════════════════════════════


class SpecEngine(Engine, ABC):
    """Shared infrastructure for speculative-decoding engines.

    Subclasses MUST:

    * Set ``self.K`` in ``__init__`` (after :meth:`Engine.__init__`).
      ``K`` is the number of drafts handed to target verify each
      iter (verify input length per req = ``K + 1``).
    * Construct their own draft model + ``draft_ctx`` (with its own
      :class:`AttnBackend`).  The draft KV cache should be sized
      with ``num_pages = target.num_pages + 1`` so target-side page
      IDs are usable interchangeably as draft slots (mirrored layout).
    * Implement :meth:`_run_prefill` and :meth:`_run_decode_iter`.

    Subclasses MAY override:

    * :meth:`_extra_mamba_checkpoints_per_req` if the spec method
      needs more than ``K`` per req for verify rollback (default ``K``).

    The :class:`Scheduler` calls :meth:`run_iter` once per scheduled
    batch and consumes :class:`SpecForwardOutput`.
    """

    K: int  # set by subclass.

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        # Injected by the scheduler after construction so the engine
        # can alloc / free target KV pages without bouncing through
        # the scheduler on every sub-forward.
        self._cache_manager: "CacheManager | None" = None

    # Override the base hook so :meth:`Engine._create_mamba_pool` sizes
    # the mamba pool to hold ``K`` extra checkpoint slots per running
    # req (used by ``forward_verify``).
    def _extra_mamba_checkpoints_per_req(self, config: EngineConfig) -> int:
        # ``self.K`` is set in subclass __init__, which runs AFTER
        # this method is invoked (from Engine.__init__).  Subclasses
        # whose K depends on config should override this method.
        return 0

    def set_cache_manager(self, cache_manager: "CacheManager") -> None:
        self._cache_manager = cache_manager

    @property
    def cache_manager(self) -> "CacheManager":
        assert self._cache_manager is not None, (
            f"{type(self).__name__}.cache_manager not set. "
            "Call set_cache_manager(...) after constructing the engine."
        )
        return self._cache_manager

    # ════════════════════════════════════════════════════════════════
    # Public entry point — called once per scheduled batch.
    # ════════════════════════════════════════════════════════════════

    def run_iter(self, batch: Batch) -> SpecForwardOutput:
        """Dispatch on batch phase."""
        if batch.is_prefill:
            return self._run_prefill(batch)
        if batch.is_decode:
            return self._run_decode_iter(batch)
        raise ValueError(
            f"Unsupported phase for {type(self).__name__}: {batch.phase}"
        )

    @abstractmethod
    def _run_prefill(self, batch: Batch) -> SpecForwardOutput:
        """Run the prompt-processing pass for ``batch``.

        Responsibilities:
        * Run target prefill (including any per-spec hidden capture).
        * Sample the first generated token ``T_1`` and commit it to
          each req's state (``input_ids``, ``cached_len``, etc.).
        * Cache whatever pending state the next decode iter needs
          on the ``Req`` (``pending_token`` + spec-specific fields).
        """

    @abstractmethod
    def _run_decode_iter(self, batch: Batch) -> SpecForwardOutput:
        """Run one decode iter (draft + verify + commit)."""

    # ════════════════════════════════════════════════════════════════
    # Shared per-iter helpers
    # ════════════════════════════════════════════════════════════════

    def _allocate_iter_slabs(
        self, reqs: List[Req],
    ) -> Tuple[mx.array, mx.array]:
        """Allocate ``B*(K+1)`` shared pages and stamp the page table.

        For each req ``i``, we reserve a contiguous ``K+1`` column
        ``slabs[i]`` of page IDs and write
        ``page_table[r, cached_len : cached_len + K + 1] = slabs[i]``.
        The SAME page IDs are used by both target verify (K+1 target
        K/V writes) and the draft (per-spec writes via mirrored
        layout).  Unused columns are released after the iter via
        :meth:`_release_iter_resources`.

        Returns:
            ``(flat_pages, slabs)`` where ``flat_pages`` is the raw
            ``[B*(K+1)]`` int32 array from :class:`CacheManager` and
            ``slabs = flat_pages.reshape(B, K+1)``.
        """
        B = len(reqs)
        verify_len = self.K + 1
        flat = self.cache_manager.allocate(B * verify_len)
        slabs = flat.reshape(B, verify_len)
        for i, r in enumerate(reqs):
            c = r.cached_len
            self.page_table[r.table_idx, c : c + verify_len] = slabs[i]
        return flat, slabs

    def _build_target_verify_batch(
        self,
        reqs: List[Req],
        verify_input_ids: mx.array,
        out_loc: mx.array,
    ) -> Tuple[Batch, List[List[int]], List[int]]:
        """Build a TARGET_VERIFY batch over ``K + 1`` positions per req.

        Allocates ``B*K`` mamba checkpoint slots so the GDN backend
        can save per-token state for rollback.  Caller is responsible
        for freeing them via :meth:`_release_iter_resources` after
        the spec-specific draft calibration (if any) has finished
        reading the verify outputs.

        Returns
            ``(batch, slot_ids_rows, new_mamba_slots)``

            * ``batch``: TARGET_VERIFY-phase batch wired with the
              target's attn metadata, ``out_loc``, and (for hybrid)
              ``mamba_slot_ids``.
            * ``slot_ids_rows``: host-side list-of-lists mirror of
              ``batch.mamba_slot_ids`` — used to compute per-req
              rollback sources without a GPU→CPU sync.
            * ``new_mamba_slots``: flat list of ``B*K`` checkpoint
              slots allocated this iter.
        """
        K = self.K
        B = len(reqs)
        verify_len = K + 1
        extend_lens = [verify_len] * B
        kv_lens = [r.cached_len + verify_len for r in reqs]
        metadata = self.attn_backend.build_prefill_metadata(
            reqs, extend_lens=extend_lens, kv_lens=kv_lens,
        )

        batch = Batch(reqs=reqs, phase=BatchPhase.TARGET_VERIFY)
        batch.input_ids = verify_input_ids
        batch.out_loc = out_loc
        batch.padded_reqs = reqs
        batch.attn_metadata = metadata

        slot_ids_rows: List[List[int]] = []
        new_mamba_slots: List[int] = []
        if self.is_hybrid:
            assert self.mamba_pool is not None
            allocated = self.mamba_pool.alloc_many(B * K)
            if allocated is None:
                raise RuntimeError(
                    f"Out of mamba checkpoint slots: needed {B * K}, "
                    f"have {self.mamba_pool.available_size}"
                )
            new_mamba_slots = allocated
            slot_ids_rows = [
                [r.mamba_slot] + new_mamba_slots[i * K : (i + 1) * K]  # type: ignore[list-item]
                for i, r in enumerate(reqs)
            ]
            batch.mamba_slot_ids = mx.array(slot_ids_rows, dtype=mx.int32)
        return batch, slot_ids_rows, new_mamba_slots

    def _release_iter_resources(
        self,
        reqs: List[Req],
        num_accepted_host: List[int],
        slabs: mx.array,
        slot_ids_rows: List[List[int]],
        new_mamba_slots: List[int],
    ) -> None:
        """Free per-iter mamba checkpoint slots + unused KV pages.

        Should be called AFTER any spec-method-specific work that
        still reads the verify outputs (e.g. EAGLE's calibration
        prefill, DFlash's bonus projection), since both pages and
        mamba state must remain live until then.

        Per req ``i`` (``j_i = num_drafts_accepted``):

        * **Mamba** (hybrid only).  ``slot_ids[i, 0]`` is the req's
          main slot — verify clobbered it with "state after the
          first verify input".  The correct post-iter state lives in
          ``slot_ids[i, j_i]`` and is copied back; all ``B*K`` new
          checkpoint slots are then returned to the pool.
        * **KV pages**.  Keep ``slabs[i, 0..j_i]`` (``j_i + 1``
          pages = accepted prefix, shared by target and draft via
          the mirrored layout); free ``slabs[i, j_i+1..K]``
          (``K - j_i`` pages, both target and draft K/V reclaimed
          together).
        """
        K = self.K
        if self.is_hybrid:
            src_slots: List[int] = []
            dst_slots: List[int] = []
            for i, r in enumerate(reqs):
                j = num_accepted_host[i]
                if j >= 1:
                    src_slots.append(slot_ids_rows[i][j])
                    dst_slots.append(r.mamba_slot)  # type: ignore[arg-type]
            if src_slots:
                assert self.mamba_pool is not None
                self.mamba_pool.copy_batched(
                    mx.array(src_slots, dtype=mx.int32),
                    mx.array(dst_slots, dtype=mx.int32),
                )
            assert self.mamba_pool is not None
            self.mamba_pool.free_many(new_mamba_slots)

        free_chunks: List[mx.array] = []
        for i, j in enumerate(num_accepted_host):
            if K - j > 0:
                free_chunks.append(slabs[i, j + 1 :])
        if free_chunks:
            self.cache_manager._free(mx.concatenate(free_chunks))

    def _prepare_target_prefill_inplace(self, batch: Batch) -> None:
        """In-place setup of a target prefill batch.

        Mirrors what :meth:`Scheduler._prepare_batch` does for the
        non-spec engine — allocate target pages, write page_table,
        concatenate input_ids and build attn / mamba metadata.  The
        spec engine prefill path goes through here directly so it
        owns the full prefill timeline (target prefill + any
        downstream draft work).
        """
        reqs = batch.reqs
        needed = sum(r.extend_len for r in reqs)
        out_loc = self.cache_manager.allocate(needed)
        batch.out_loc = out_loc
        batch.padded_reqs = reqs

        offset = 0
        chunks: List[mx.array] = []
        for req in reqs:
            n = req.extend_len
            if n > 0:
                self.page_table[
                    req.table_idx, req.cached_len : req.device_len
                ] = out_loc[offset : offset + n]
                chunks.append(req.input_ids[req.cached_len : req.device_len])
                offset += n
        batch.input_ids = (
            mx.concatenate(chunks) if chunks else mx.array([], dtype=mx.int32)
        )
        self.attn_backend.prepare_metadata(batch)
        if self.gdn_backend is not None:
            self.gdn_backend.prepare_batch(batch)
