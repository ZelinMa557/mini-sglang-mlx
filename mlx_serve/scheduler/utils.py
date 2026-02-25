from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_serve.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: mx.array
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[mx.array]
