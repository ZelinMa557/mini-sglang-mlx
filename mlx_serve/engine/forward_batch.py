import mlx.core as mx
from dataclasses import dataclass
from enum import Enum


class ForwardType(Enum):
    none = 0
    prefill = 1
    decode = 2


@dataclass
class ForwardBatch:
    request_ids: list[int] = None
    seq_lens: mx.array = None
    offsets: mx.array = None
    position_ids: mx.array = None
    last_positions: mx.array = None
    forward_type: ForwardType = ForwardType.none
    temperatures: mx.array = None
    top_ks: mx.array = None
    top_ps: mx.array = None
    input_ids: mx.array = None
    scheduled_requests: list = None

    def __init__(self, request_ids: list[int], seq_lens: mx.array, offsets: mx.array, forward_type: ForwardType, temperatures: mx.array = None, top_ks: mx.array = None, top_ps: mx.array = None, input_ids: mx.array = None, scheduled_requests: list = None):
        self.request_ids = request_ids
        self.seq_lens = seq_lens
        self.offsets = offsets
        self.forward_type = forward_type
        self.temperatures = temperatures
        self.top_ks = top_ks
        self.top_ps = top_ps
        self.input_ids = input_ids
        self.scheduled_requests = scheduled_requests or []
        if seq_lens is not None and len(seq_lens) > 0:
            self._init_position_ids()
            self._init_last_positions()


    def _init_position_ids(self):
        batch_size = len(self.seq_lens)
        if batch_size == 1:
            self.position_ids = (
                mx.arange(self.seq_lens[0], dtype=mx.int32) + self.offsets[0]
            )
            return
        self.position_ids = mx.concatenate(
            [
                mx.arange(seq_len) + offset
                for seq_len, offset in zip(self.seq_lens, self.offsets)
            ]
        )
    
    def _init_last_positions(self):
        shifted_seq_lens = mx.concatenate([self.seq_lens[1:], mx.array([0], dtype=self.seq_lens.dtype)])
        self.last_positions = mx.cumsum(shifted_seq_lens) - 1
