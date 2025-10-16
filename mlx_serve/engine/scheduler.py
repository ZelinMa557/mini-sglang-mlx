from mlx_serve.engine.forward_batch import ForwardBatch, ForwardType
from mlx_serve.engine.forward_request import ForwardRequest, ForwardRequestPool
from dataclasses import dataclass
@dataclass
class SchedulerConfig:
    max_running_tokens: int
    max_ongoing_requests: int

class Scheduler:
    def __init__(self, config: SchedulerConfig, request_pool: ForwardRequestPool):
        self.config = config

        self.running_queue: list[int] = []
        self.waiting_queue: list[int] = []

        self.request_pool = request_pool
        pass

    def next_batch(self) -> ForwardBatch:
        