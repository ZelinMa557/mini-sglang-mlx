from dataclasses import dataclass

@dataclass
class ForwardRequest:
    input_tokens: list[int] = None
    generated_tokens: list[int] = None

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
        