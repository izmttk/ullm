import torch
from ..distributed.communication_op import send_tensor_dict, recv_tensor_dict
from ..distributed.parallel_state import (
    get_world_group,
    get_tp_group,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
    destroy_distributed_environment,
    destroy_model_parallel,
)
from .model_runner import ModelRunner
from .common import ForwardBatch
from ..layers.utils import IntermediateTensors

class Worker:
    def __init__(
        self,
        model: str,
        max_bs: int,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        enforce_eager: bool = False,
        context_len: int = 2048,
    ):
        self.model = model
        self.max_bs = max_bs
        
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nccl_port = nccl_port
        self.enforce_eager = enforce_eager
        self.context_len = context_len
        
        self.rank = pp_rank * tp_size + tp_rank
        self.world_size = pp_size * tp_size


    def init_environment(self):
        init_distributed_environment(
            word_size=self.world_size,
            rank=self.rank,
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{self.nccl_port}",
        )
        initialize_model_parallel(
            self.tp_size,
            self.pp_size,
            self.tp_rank,
            self.pp_rank,
        )

        self.device = torch.device(f"cuda:{self.rank}")
        self.model_runner = ModelRunner(
            model=self.model,
            max_bs=self.max_bs,
            rank=self.rank,
            device=self.device,
            enforce_eager=self.enforce_eager,
            context_len=self.context_len,
        )
        print(f"Worker {self.rank} started with TP rank {self.tp_rank}, PP rank {self.pp_rank}.")

    def destroy_environment(self):
        # Resolve hanging issues with CUDA graphs enabled
        # See: https://github.com/pytorch/pytorch/issues/115388
        get_world_group().barrier()
        torch.cuda.synchronize()
        if self.model_runner.cuda_graph is not None:
            self.model_runner.cuda_graph.clear()

        destroy_model_parallel()
        destroy_distributed_environment()
        print(f"Worker {self.rank} destroyed its environment.")

    def load_model(self):
        self.model_runner.load_model()

    def execute_model(self, batch: ForwardBatch) -> list[int] | None:
        intermediate_tensors = None
        
        if not get_pp_group().is_first_rank:
            # print(f"Worker {self.rank} waiting to receive intermediate tensors.")
            recv = recv_tensor_dict(group=get_pp_group(), all_gather_group=get_tp_group())
            # print(f"Worker {self.rank} received intermediate tensors.")
            assert recv is not None
            intermediate_tensors = IntermediateTensors(recv)
        # print(f"Worker {self.rank} executing model.")
        output = self.model_runner.execute_model(batch, intermediate_tensors)
        # print(f"Worker {self.rank} finished model execution.")
        if not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            # print(f"Worker {self.rank} sending intermediate tensors to next PP rank.")
            send_tensor_dict(output, group=get_pp_group(), all_gather_group=get_tp_group())
            # print(f"Worker {self.rank} sent intermediate tensors.")
            return None
        
        assert isinstance(output, torch.Tensor)
        return output.tolist()

    def initialize_kv_cache(self, kv_cache_size: int):
        self.model_runner.initialize_kv_cache(kv_cache_size)
    
    def profile_kv_cache_size(self, gpu_memory_utilization: float):
        return self.model_runner.profile_kv_cache_size(gpu_memory_utilization)
