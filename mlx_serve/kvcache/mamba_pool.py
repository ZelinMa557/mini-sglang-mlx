from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple

import mlx.core as mx

from mlx_serve.utils import init_logger

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
    dtype: mx.Dtype = mx.float16


class MambaStatePool:
    """Slot-based pool for Mamba conv + temporal (SSM) state across all layers.

    Each slot holds the full recurrent state for one request (or one cached
    prefix in the radix tree).  Slot 0 is reserved as a dummy/padding slot.
    """

    def __init__(self, config: MambaStateConfig) -> None:
        self.num_slots = config.num_slots
        self.num_layers = config.num_layers
        self.dtype = config.dtype

        self._conv_buffers: List[List[mx.array]] = []
        self._temporal_buffers: List[mx.array] = []

        for layer_idx in range(config.num_layers):
            layer_convs: List[mx.array] = []
            for conv_shape in config.conv_shapes[layer_idx]:
                buf = mx.zeros(
                    (config.num_slots + 1, *conv_shape), dtype=config.dtype
                )
                layer_convs.append(buf)
            self._conv_buffers.append(layer_convs)

            temporal_buf = mx.zeros(
                (config.num_slots + 1, *config.temporal_shapes[layer_idx]),
                dtype=config.dtype,
            )
            self._temporal_buffers.append(temporal_buf)

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
        return bufs

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def alloc(self) -> int | None:
        if not self._free_slots:
            return None
        slot = self._free_slots.pop()
        self._zero_slot(slot)
        return slot

    def free(self, slot: int) -> None:
        assert 1 <= slot <= self.num_slots, f"Invalid slot {slot}"
        self._free_slots.append(slot)

    def copy(self, src: int, dst: int) -> None:
        """Copy all mamba state from *src* slot to *dst* slot."""
        for layer_convs in self._conv_buffers:
            for buf in layer_convs:
                buf[dst] = buf[src]
        for buf in self._temporal_buffers:
            buf[dst] = buf[src]

    def conv_state(self, layer_idx: int, group: int = 0) -> mx.array:
        return self._conv_buffers[layer_idx][group]

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
