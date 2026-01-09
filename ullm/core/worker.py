import os
from pathlib import Path

import torch

from ..config import EngineConfig
from ..distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    recv_tensor_dict,
    send_tensor_dict,
)
from ..layers.utils import IntermediateTensors
from ..logger import init_logger
from .common import ForwardBatch, ForwardMode, SchedulerOutput, Sequence
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
        self.config = config

        self.rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.nccl_port = config.nccl_port

        self.world_size = self.tp_size * self.pp_size

        self.sequences: dict[str, Sequence] = {}

        if config.profile:
            profile_dir = Path(config.profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_dir = str(profile_dir.resolve())
            self.profile_dir = profile_dir
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profile_dir,
                    worker_name=f"ullm_worker_{self.rank}_TP_{self.tp_rank}_PP_{self.pp_rank}_{os.getpid()}",
                    use_gzip=True,
                ),
                record_shapes=True,
                with_stack=True,
            )
            self.profiler_started = False

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
            config=self.config,
            rank=self.rank,
            device=self.device,
        )
        logger.debug(
            f"Worker {self.rank} started with TP rank {self.tp_rank}, PP rank {self.pp_rank}."
        )

    def destroy_environment(self):
        # Resolve hanging issues with CUDA graphs enabled
        # See: https://github.com/pytorch/pytorch/issues/115388
        torch.cuda.synchronize()
        if (
            hasattr(self, "model_runner")
            and hasattr(self.model_runner, "cuda_graph")
            and self.model_runner.cuda_graph is not None
        ):
            self.model_runner.cuda_graph.clear()

        destroy_model_parallel()
        destroy_distributed_environment()
        logger.debug(f"Worker {self.rank} destroyed its environment.")

    def initialize(self, gpu_memory_utilization: float):
        self.model_runner.initialize(gpu_memory_utilization)

    def apply_sequence_updates(self, sched_output: SchedulerOutput) -> ForwardBatch:
        for finished_seq_id in sched_output.finished_seq_ids:
            if finished_seq_id in self.sequences:
                del self.sequences[finished_seq_id]
        for preempted_seq_id in sched_output.preempted_seq_ids:
            if preempted_seq_id in self.sequences:
                del self.sequences[preempted_seq_id]

        prefill_batch = []
        for scheduled_new_seq in sched_output.scheduled_new_seqs:
            seq = scheduled_new_seq.to_sequence()
            self.sequences[seq.seq_id] = seq
            prefill_batch.append(seq)
        decode_batch = []
        for scheduled_cached_seq in sched_output.scheduled_cached_seqs:
            seq = self.sequences[scheduled_cached_seq.seq_id]
            scheduled_cached_seq.apply_updates(seq)
            decode_batch.append(seq)

        assert not (prefill_batch and decode_batch), (
            "Cannot have both prefill and decode batches."
        )
        if prefill_batch:
            return ForwardBatch(
                forward_mode=ForwardMode.PREFILL,
                num_seqs=len(prefill_batch),
                seqs=prefill_batch,
            )
        else:
            return ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                num_seqs=len(decode_batch),
                seqs=decode_batch,
            )

    def execute_model(self, sched_output: SchedulerOutput) -> list[int] | None:
        intermediate_tensors = None
        batch = self.apply_sequence_updates(sched_output)

        if hasattr(self, "profiler") and self.profiler_started:
            self.profiler.step()

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

    def profile(self, action: str):
        if hasattr(self, "profiler"):
            if action == "start":
                if self.profiler_started:
                    logger.warning(
                        f"Profiler already started on Worker {self.rank}, ignoring."
                    )
                else:
                    self.profiler.start()
                    self.profiler_started = True
                    logger.debug(f"Profiler started on Worker {self.rank}.")
            elif action == "stop":
                if not self.profiler_started:
                    logger.warning(
                        f"Profiler already stopped on Worker {self.rank}, ignoring."
                    )
                else:
                    self.profiler.stop()
                    self.profiler_started = False
                    logger.debug(f"Profiler stopped on Worker {self.rank}.")
            else:
                logger.warning(f"Unknown profiling action: {action}, ignoring.")
        else:
            logger.warning(f"Profiler not enabled on Worker {self.rank}.")
