import torch

from ..config import EngineConfig
from ..distributed.communication_op import recv_tensor_dict, send_tensor_dict
from ..distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from ..layers.utils import IntermediateTensors
from ..logger import init_logger
from .common import ForwardBatch
from .model_runner import ModelRunner

logger = init_logger(__name__)


class Worker:
    def __init__(
        self,
        config: EngineConfig,
        rank: int,
        tp_rank: int,
        pp_rank: int,
    ):
        self.model = config.model
        self.max_bs = config.max_bs

        self.rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.nccl_port = config.nccl_port
        self.enforce_eager = config.enforce_eager
        self.context_len = config.context_len

        self.world_size = self.tp_size * self.pp_size

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
        logger.debug(
            f"Worker {self.rank} started with TP rank {self.tp_rank}, PP rank {self.pp_rank}."
        )

    def destroy_environment(self):
        # Resolve hanging issues with CUDA graphs enabled
        # See: https://github.com/pytorch/pytorch/issues/115388
        torch.cuda.synchronize()
        if (
            hasattr(self.model_runner, "cuda_graph")
            and self.model_runner.cuda_graph is not None
        ):
            self.model_runner.cuda_graph.clear()

        destroy_model_parallel()
        destroy_distributed_environment()
        logger.debug(f"Worker {self.rank} destroyed its environment.")

    def initialize(self, gpu_memory_utilization: float):
        self.model_runner.initialize(gpu_memory_utilization)

    def execute_model(self, batch: ForwardBatch) -> list[int] | None:
        intermediate_tensors = None

        if not get_pp_group().is_first_rank:
            recv = recv_tensor_dict(
                group=get_pp_group(), all_gather_group=get_tp_group()
            )
            assert recv is not None
            intermediate_tensors = IntermediateTensors(recv)
        output = self.model_runner.execute_model(batch, intermediate_tensors)
        if not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            send_tensor_dict(
                output, group=get_pp_group(), all_gather_group=get_tp_group()
            )
            return None

        assert isinstance(output, torch.Tensor)
        return output.tolist()

    def get_kv_cache_size(self) -> int:
        assert hasattr(self.model_runner, "kv_cache_size"), (
            "Initialize Worker before calling this function."
        )
        return self.model_runner.kv_cache_size
