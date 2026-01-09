import bisect
from dataclasses import fields, is_dataclass
from typing import Any, Callable

import torch
from torch.utils._pytree import tree_flatten


class CUDAGraphRunner:
    def __init__(
        self,
        mempool: Any | None = None,
        num_warmup: int = 3,
    ):
        self.mempool = mempool
        self.num_warmup = num_warmup
        self.input_tensors = None
        self.output_tensors = None
        self.outputs = None
        self.graph = None

    def get_tensors(self, *args, **kwargs):
        tensors = []
        args, _ = tree_flatten((args, kwargs))
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensors.append(arg)
            if is_dataclass(arg) and not isinstance(arg, type):
                tensors.extend(
                    self.get_tensors([getattr(arg, f.name) for f in fields(arg)])
                )
        return tensors

    def apply_input_tensors(self, *args, **kwargs):
        if self.input_tensors is None:
            raise ValueError("Input tensors have not been set.")

        new_tensors = self.get_tensors(*args, **kwargs)
        for src, dst in zip(new_tensors, self.input_tensors):
            if src.data_ptr() != dst.data_ptr():
                dst.copy_(src)

    def capture(self, model: Callable, args=(), kwargs={}, context=None):
        self.graph = torch.cuda.CUDAGraph()
        self.mempool = self.graph.pool() if self.mempool is None else self.mempool

        self.input_tensors = self.get_tensors(args, kwargs, context)

        # warm up
        for _ in range(self.num_warmup):
            model(*args, **kwargs)

        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, pool=self.mempool):
            outputs = model(*args, **kwargs)

        self.outputs = outputs
        self.output_tensors = self.get_tensors(outputs)
        return outputs

    def replay(self, args=(), kwargs={}, context=None):
        assert self.graph is not None, "CUDAGraph has not been captured yet."
        self.apply_input_tensors(args, kwargs, context)
        self.graph.replay()
        return self.outputs


class CUDAGraphManager:
    def __init__(self):
        self.graphs: dict[int, CUDAGraphRunner] = {}
        self.graph_keys = []
        self.mempool = torch.cuda.graph_pool_handle()

    def get_graph_runner(self, bs: int) -> tuple[int, CUDAGraphRunner]:
        if bs in self.graphs:
            return bs, self.graphs[bs]
        else:
            # Find the closest larger batch size
            available_idx = bisect.bisect_left(self.graph_keys, bs)
            if available_idx < len(self.graph_keys):
                bs = self.graph_keys[available_idx]
                return bs, self.graphs[bs]
            else:
                raise ValueError(
                    f"No CUDAGraphRunner available for batch size {bs} or larger."
                )

    def create_graph_runner(self, bs: int) -> CUDAGraphRunner:
        graph_runner = CUDAGraphRunner(mempool=self.mempool)
        self.graphs[bs] = graph_runner
        self.graph_keys.append(bs)
        self.graph_keys.sort()
        return graph_runner

    def clear(self):
        self.graphs.clear()
        self.graph_keys.clear()
