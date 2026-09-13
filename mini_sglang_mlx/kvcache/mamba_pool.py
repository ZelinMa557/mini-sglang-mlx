from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, NamedTuple

import mlx.core as mx

from mini_sglang_mlx.utils import init_logger

logger = init_logger(__name__)


class MambaStateShape(NamedTuple):
    """Shape descriptors for a single Mamba layer's state buffers."""

    conv: List[tuple]  # list of (d_inner, d_conv) tuples — one per conv group
    temporal: tuple  # (num_heads, head_dim, state_size) or similar


@dataclass(frozen=True)
class MambaStateConfig:
    """Everything needed to allocate the pool."""

    num_slots: int
    num_layers: int
    conv_shapes: List[List[tuple]]  # per-layer list of conv shapes
    temporal_shapes: List[tuple]  # per-layer temporal shape
    dtype: mx.Dtype = mx.float32
    # Spec-decoding verify window width (K + 1).  When non-zero the
    # pool also allocates per-layer conv window buffers — each slot
    # holds ``verify_width`` sliding windows, one per verify step, so
    # target verify can checkpoint the tiny conv state per step
    # without allocating full (conv + temporal) slots.
    verify_width: int = 0


class MambaStatePool:
    """Slot-based pool for Mamba conv + temporal (SSM) state across all layers.

    Each slot holds the full recurrent state for one request (or one cached
    prefix in the radix tree).  Slot 0 is reserved as a dummy/padding slot.
    """

    def __init__(self, config: MambaStateConfig) -> None:
        self.num_slots = config.num_slots
        self.num_layers = config.num_layers
        self.dtype = config.dtype
        self.verify_width = config.verify_width

        self._conv_buffers: List[List[mx.array]] = []
        self._temporal_buffers: List[mx.array] = []
        self._conv_window_buffers: List[List[mx.array]] = []

        for layer_idx in range(config.num_layers):
            layer_convs: List[mx.array] = []
            for conv_shape in config.conv_shapes[layer_idx]:
                buf = mx.zeros(
                    (config.num_slots + 1, *conv_shape), dtype=mx.bfloat16
                )
                layer_convs.append(buf)
            self._conv_buffers.append(layer_convs)

            temporal_buf = mx.zeros(
                (config.num_slots + 1, *config.temporal_shapes[layer_idx]),
                dtype=config.dtype,
            )
            self._temporal_buffers.append(temporal_buf)

            layer_windows: List[mx.array] = []
            if self.verify_width > 0:
                # conv shape convention: (state_len, d_inner).
                for state_len, d_inner in config.conv_shapes[layer_idx]:
                    layer_windows.append(
                        mx.zeros(
                            (config.num_slots + 1, self.verify_width,
                             state_len, d_inner),
                            dtype=mx.bfloat16,
                        )
                    )
            self._conv_window_buffers.append(layer_windows)

        mx.eval(*self._all_buffers())

        self._free_slots: List[int] = list(range(1, config.num_slots + 1))

        total_bytes = sum(b.nbytes for b in self._all_buffers())
        logger.info(
            "MambaStatePool: %d slots, %d layers, %.2f MB",
            config.num_slots,
            config.num_layers,
            total_bytes / (1024 * 1024),
        )

    def _all_buffers(self) -> List[mx.array]:
        bufs: List[mx.array] = []
        for layer_convs in self._conv_buffers:
            bufs.extend(layer_convs)
        bufs.extend(self._temporal_buffers)
        for layer_windows in self._conv_window_buffers:
            bufs.extend(layer_windows)
        return bufs

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def check_ownership(self, tree_slots: Iterable[int]) -> None:
        """Assert every slot is owned by the free list or by a cached prefix.

        A slot has exactly one owner at a time: this pool's free list, a
        radix-tree node holding a snapshot, or a live request.  This checks
        the first two, so it is only meaningful when no request is running --
        which is precisely when the scheduler calls it (see
        ``Scheduler.run_when_idle``).  A slot owned by neither is leaked:
        nothing will ever return it, and the pool shrinks by one forever.
        """
        owned = set(self._free_slots)
        assert len(owned) == len(
            self._free_slots
        ), "free list contains duplicates"
        for slot in tree_slots:
            assert 1 <= slot <= self.num_slots, f"tree holds invalid slot {slot}"
            assert slot not in owned, f"Slot {slot} is owned twice"
            owned.add(slot)
        leaked = set(range(1, self.num_slots + 1)) - owned
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} mamba slot(s) leaked: {sorted(leaked)} are "
                "neither free nor held by a cached prefix"
            )

    def alloc(self) -> int | None:
        if not self._free_slots:
            return None
        slot = self._free_slots.pop()
        self._zero_slot(slot)
        return slot

    def alloc_many(self, n: int) -> List[int] | None:
        """Allocate ``n`` slots in one Python call (no zero-init).

        Slots are returned in arbitrary order. Caller is responsible for
        zeroing if needed — for verify scratch slots we don't need to
        zero because the engine copies the main slot into each scratch
        slot before the verify forward reads it.
        """
        if n == 0:
            return []
        if len(self._free_slots) < n:
            return None
        slots = self._free_slots[-n:]
        del self._free_slots[-n:]
        return slots

    def free(self, slot: int) -> None:
        assert 1 <= slot <= self.num_slots, f"Invalid slot {slot}"
        # A slot freed twice would sit in the free list twice and could be
        # handed to two sequences at once, i.e. they would silently share one
        # recurrent state.  There is no refcount to catch that later, so it
        # has to fail here.
        assert slot not in self._free_slots, f"Slot {slot} is already free"
        self._free_slots.append(slot)

    def free_many(self, slots: List[int]) -> None:
        """Return ``slots`` to the free pool in one Python call."""
        for slot in slots:
            self.free(slot)

    def copy(self, src: int, dst: int) -> None:
        """Copy all mamba state from *src* slot to *dst* slot."""
        for layer_convs in self._conv_buffers:
            for buf in layer_convs:
                buf[dst] = buf[src]
        for buf in self._temporal_buffers:
            buf[dst] = buf[src]

    def copy_batched(self, src: mx.array, dst: mx.array) -> None:
        """Batched scatter-copy: for each i, copy slot src[i] -> slot dst[i].

        ``src`` and ``dst`` are int32 arrays of the same shape.  Used by
        replay-style target verify to seed each req's scratch slot with
        a copy of its main slot before the verify window is replayed.
        """
        assert src.shape == dst.shape
        for layer_convs in self._conv_buffers:
            for buf in layer_convs:
                buf[dst] = buf[src]
        for buf in self._temporal_buffers:
            buf[dst] = buf[src]

    def conv_state(self, layer_idx: int, group: int = 0) -> mx.array:
        return self._conv_buffers[layer_idx][group]

    def conv_window(self, layer_idx: int, group: int = 0) -> mx.array:
        """Per-step conv sliding windows used by replay-style verify.

        Row ``[slot, j]`` holds the conv window *after* processing
        ``j + 1`` verify tokens.  Only allocated when
        :attr:`MambaStateConfig.verify_width` is non-zero.
        """
        assert self.verify_width > 0, "conv windows not allocated"
        return self._conv_window_buffers[layer_idx][group]

    def temporal_state(self, layer_idx: int) -> mx.array:
        return self._temporal_buffers[layer_idx]

    def _zero_slot(self, slot: int) -> None:
        for layer_convs in self._conv_buffers:
            for buf in layer_convs:
                buf[slot] = mx.zeros(buf.shape[1:], dtype=self.dtype)
        for buf in self._temporal_buffers:
            buf[slot] = mx.zeros(buf.shape[1:], dtype=self.dtype)

    def mem_usage_bytes(self) -> int:
        return sum(b.nbytes for b in self._all_buffers())
