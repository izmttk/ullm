import os
import queue
from concurrent.futures import Future
from pathlib import Path
from typing import Callable

import torch

from ..config import EngineConfig
from ..logger import init_logger
from .common import (
    EngineStepResult,
    FinishReason,
    ForwardBatch,
    SamplingParams,
    Sequence,
)
from .executor import Executor
from .scheduler import Scheduler

logger = init_logger(__name__)


class Engine:
    """
    The Engine class coordinates the scheduler and the executor to run the LLM inference.
    """

    def __init__(
        self,
        config: EngineConfig,
        failure_callback: Callable[[], None] | None = None,
    ):
        self.config = config
        self.model_executor = Executor(config=self.config)
        if failure_callback:
            self.model_executor.on_failure(failure_callback)
        self.model_executor.initialize()
        kv_cache_size = self.model_executor.get_kv_cache_size()
        self.scheduler = Scheduler(
            config=self.config,
            kv_cache_size=kv_cache_size,
        )

        self.pp_queue: queue.Queue[tuple[Future[list[int]], ForwardBatch]] | None = None
        if self.config.pp_size > 1:
            self.pp_queue = queue.Queue(self.config.pp_size)

        if config.profile:
            profile_dir = Path(config.profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_dir = str(profile_dir.resolve())
            self.profile_dir = profile_dir
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profile_dir,
                    worker_name=f"ullm_engine_{os.getpid()}",
                    use_gzip=True,
                ),
                with_stack=True,
            )
            self.profiler_started = False

    def add_sequence(
        self,
        sequence_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ):
        """
        Add a new sequence to the engine's scheduler.
        """
        if len(prompt_token_ids) > self.config.context_len:
            prompt_token_ids = prompt_token_ids[-self.config.context_len :]

        seq = Sequence(
            seq_id=sequence_id,
            token_ids=prompt_token_ids,
            num_tokens=len(prompt_token_ids),
            prompt_len=len(prompt_token_ids),
            sampling_params=sampling_params,
        )
        self.scheduler.add_sequence(seq)
        logger.info(f"Added sequence {sequence_id}.")

    def abort_sequence(self, sequence_id: str):
        """
        Abort a sequence in the engine's scheduler.
        """
        seq = self.scheduler.get_sequence(sequence_id)
        if seq:
            self.scheduler.finish_sequence(seq)
        logger.info(f"Aborted sequence {sequence_id}.")

    def step(self) -> list[EngineStepResult]:
        """
        Performs one step of inference.

        1. Schedules a batch of sequences.
        3. Executes the model.
        5. Updates the sequences.
        """
        if self.pp_queue is not None:
            # 注意当开启了流水线并行时，step 可能会返回空列表
            return self.step_with_pp()

        batch = self.scheduler.schedule()
        if not batch:
            return []
        fut = self.model_executor.execute_model(batch)
        output_ids = fut.result()
        outputs = self.update_from_output(batch, output_ids)
        return outputs

    def step_with_pp(self) -> list[EngineStepResult]:
        assert self.pp_queue is not None
        outputs: list[EngineStepResult] = []
        batch = None
        if not self.pp_queue.full():
            batch = self.scheduler.schedule()
            if batch:
                fut = self.model_executor.execute_model(batch)
                self.pp_queue.put_nowait((fut, batch))
        # 如果 batch 为 None，说明要么没有新请求，要么 pp 队列已满不允许调度
        if batch is None and not self.pp_queue.empty():
            (fut, sched_batch) = self.pp_queue.get_nowait()
            output_ids = fut.result()  # 阻塞等待结果
            outputs = self.update_from_output(sched_batch, output_ids)

        return outputs

    def update_from_output(self, batch: ForwardBatch, output_ids: list[int]):
        """
        Update sequences based on the model output.
        """
        outputs: list[EngineStepResult] = []
        # Update sequences with the model output
        for seq, new_token_id in zip(batch.seqs, output_ids):
            self.scheduler.update_sequence(seq, new_token_id)

            is_finished, finish_reason = self._is_sequence_finished(seq)
            if is_finished:
                self.scheduler.finish_sequence(seq)

            outputs.append(
                EngineStepResult(
                    seq_id=seq.seq_id,
                    new_token_id=new_token_id,
                    is_finished=is_finished,
                    finish_reason=finish_reason,
                    num_prompt_tokens=seq.prompt_len,
                    num_generated_tokens=seq.num_tokens - seq.prompt_len,
                )
            )

        return outputs

    def _is_sequence_finished(self, seq: Sequence) -> tuple[bool, FinishReason | None]:
        # Check for stop tokens
        if (
            seq.token_ids[-1] == seq.sampling_params.eos_token_id
            and not seq.sampling_params.ignore_eos
        ):
            return True, FinishReason.STOP

        # Check for max tokens
        if (
            seq.sampling_params.max_tokens
            and seq.num_tokens >= seq.sampling_params.max_tokens
        ):
            return True, FinishReason.LENGTH
        if (
            seq.sampling_params.max_new_tokens
            and seq.num_tokens >= seq.prompt_len + seq.sampling_params.max_new_tokens
        ):
            return True, FinishReason.LENGTH

        return False, None

    def shutdown(self):
        logger.debug(f"Shutting down {self.__class__.__name__}...")
        self.model_executor.shutdown()
        logger.debug(f"{self.__class__.__name__} shut down.")

    def has_unfinished_sequences(self):
        return self.scheduler.has_unfinished_sequences()

    def profile(self, action: str):
        if hasattr(self, "profiler"):
            if action == "start":
                if self.profiler_started:
                    logger.warning("Profiler already started on Engine, ignoring.")
                else:
                    self.profiler.start()
                    self.profiler_started = True
                    logger.debug("Profiler started on Engine.")
            elif action == "stop":
                if not self.profiler_started:
                    logger.warning("Profiler already stopped on Engine, ignoring.")
                else:
                    logger.info("Stopping profiling, this may take a while...")
                    self.profiler.stop()
                    self.profiler_started = False
                    logger.debug("Profiler stopped on Engine.")
            else:
                logger.warning(f"Unknown profiling action: {action}, ignoring.")
        else:
            logger.warning("Profiler not initialized on Engine, ignoring.")
        self.model_executor.profile(action)
        if hasattr(self, "profiler"):
            if action == "start":
                logger.info("Profiler successfully started.")
            elif action == "stop":
                logger.info(
                    f"Profiler successfully stopped, data saved to {self.profile_dir}."
                )
