from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from mlx_serve.core import Batch, BatchPhase, Req


@dataclass
class DecodeManager:
    running_reqs: Set[Req] = field(default_factory=set)
    # Per-req fudge added to ``inflight_tokens`` so the prefill scheduler
    # leaves room for transient peak allocations during an iter.  Used
    # by speculative decoding, where each iter briefly holds K extra
    # target KV pages before freeing them after verify.
    extra_per_req: int = 0

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode()}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    @property
    def inflight_tokens(self) -> int:
        return sum(req.remain_len + self.extra_per_req for req in self.running_reqs)

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase=BatchPhase.DECODE)

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
