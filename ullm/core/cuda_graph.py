from contextlib import contextmanager
import torch
import bisect

class CUDAGraph:
    def __init__(self):
        self.is_captured = False
        self.max_bs = 0
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_pool = None
        self.input_buffers: dict[str, torch.Tensor] = {}
        self.output_buffers: dict[str, torch.Tensor] = {}
        self.captured_bs: list[int] = []

    def set_input_buffer(self, name: str, tensor: torch.Tensor):
        self.input_buffers[name] = tensor

    def get_input_buffer(self, name: str):
        return self.input_buffers[name]

    def set_output_buffer(self, name: str, tensor: torch.Tensor):
        self.output_buffers[name] = tensor

    def get_output_buffer(self, name: str):
        return self.output_buffers[name]

    @contextmanager
    def capture(self, bs: int):
        self.max_bs = max(self.max_bs, bs)
        graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(graph, self.graph_pool):
            yield
        
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
        bisect.insort(self.captured_bs, bs)
        self.is_captured = True

    def replay(self, bs: int):
        self.graphs[bs].replay()

    def match_bs(self, bs: int) -> int:
        assert bs <= self.max_bs and self.graphs
        index = bisect.bisect_left(self.captured_bs, bs)
        return self.captured_bs[index]
    
    def clear(self):
        self.graphs.clear()
        self.captured_bs.clear()
        self.is_captured = False
        self.max_bs = 0
        self.graph_pool = None
        self.input_buffers.clear()
        self.output_buffers.clear()
