from dataclasses import dataclass
from enum import Enum, auto
class ForwardRequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

@dataclass
class ForwardRequest:
    id: int
    input_tokens: list[int] = None
    generated_tokens: list[int] = None
    temperature: float = None
    top_k: int = None
    top_p: float = None
    status: ForwardRequestStatus = ForwardRequestStatus.WAITING

class ForwardRequestPool:
    def __init__(self, size: int):
        self.free_list: list[int] = [ _ for _ in range(size)]
        self.pool: list[ForwardRequest] = [ForwardRequest() for _ in range(size)]
        self.size = size

    def add(self, request: ForwardRequest) -> int:
        if len(self.free_list) == 0:
            return -1
        request_id = self.free_list[-1]
        self.pool[request_id] = request
        self.free_list.pop()
        return request_id
    
    def get(self, request_id: int) -> ForwardRequest:
        assert request_id < self.size
        return self.pool[request_id]
    
    def free(self, request_id: int):
        assert request_id < self.size
        self.free_list.append(request_id)
        