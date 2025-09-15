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
