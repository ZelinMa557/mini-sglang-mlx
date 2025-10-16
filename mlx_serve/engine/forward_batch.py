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
