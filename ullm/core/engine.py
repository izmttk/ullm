from concurrent.futures import Future
import queue
from .common import FinishReason, ForwardBatch, SamplingParams, Sequence, EngineOutput
from .executor import Executor
from .scheduler import Scheduler


class Engine:
    """
    The Engine class coordinates the scheduler and the executor to run the LLM inference.
    """

    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float,
        max_bs: int,
        tp_size: int,
        pp_size: int,
        nccl_port: int = 29500,
        device_ids: list[int] | None = None,
        enforce_eager: bool = False,
        context_len: int = 2048,
    ):
        self.context_len = context_len
        self.model_executor = Executor(
            model=model,
            max_bs=max_bs,
            tp_size=tp_size,
            pp_size=pp_size,
            nccl_port=nccl_port,
            device_ids=device_ids,
            enforce_eager=enforce_eager,
            context_len=context_len,
        )
        
        kv_cache_size = self.model_executor.profile_kv_cache_size(gpu_memory_utilization)
        print(f"Max num tokens in kv cache: {kv_cache_size}")
        self.model_executor.initialize_kv_cache(kv_cache_size)

        self.scheduler = Scheduler(
            kv_cache_size=kv_cache_size,
            max_bs=max_bs
        )
        
        self.pp_queue: queue.Queue[tuple[Future[list[int]], ForwardBatch]] | None = None
        if pp_size > 1:
            self.pp_queue = queue.Queue(pp_size)

    def add_sequence(
        self,
        sequence_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ):
        """
        Add a new sequence to the engine's scheduler.
        """
        if len(prompt_token_ids) > self.context_len:
            prompt_token_ids = prompt_token_ids[-self.context_len:]

        seq = Sequence(
            seq_id=sequence_id,
            token_ids=prompt_token_ids,
            num_tokens=len(prompt_token_ids),
            prompt_len=len(prompt_token_ids),
            sampling_params=sampling_params,
        )
        self.scheduler.add_sequence(seq)
        print(f"Added sequence {sequence_id}")
    
    def abort_sequence(self, sequence_id: str):
        """
        Abort a sequence in the engine's scheduler.
        """
        seq = self.scheduler.get_sequence(sequence_id)
        if seq:
            self.scheduler.finish_sequence(seq)
        print(f"Aborted sequence {sequence_id}")

    def step(self) -> list[EngineOutput]:
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
        # print(f"Scheduled batch {batch.forward_mode.name} with {batch.num_seqs} sequences.")
        fut = self.model_executor.execute_model(batch)
        output_ids = fut.result()
        outputs = self.update_from_output(batch, output_ids)
        return outputs

    def step_with_pp(self) -> list[EngineOutput]:
        assert self.pp_queue is not None
        outputs: list[EngineOutput] = []
        batch = None
        if not self.pp_queue.full():
            batch = self.scheduler.schedule()
            if batch:
                fut = self.model_executor.execute_model(batch)
                self.pp_queue.put_nowait((fut, batch))
        # 如果 batch 为 None，说明要么没有新请求，要么 pp 队列已满不允许调度
        if batch is None and not self.pp_queue.empty():
            (fut, sched_batch) = self.pp_queue.get_nowait()
            output_ids = fut.result() # 阻塞等待结果
            outputs = self.update_from_output(sched_batch, output_ids)
            
        return outputs

    def update_from_output(self, batch: ForwardBatch, output_ids: list[int]):
        """
        Update sequences based on the model output.
        """
        outputs: list[EngineOutput] = []
        # Update sequences with the model output
        for seq, new_token_id in zip(batch.seqs, output_ids):
            self.scheduler.update_sequence(seq, new_token_id)

            is_finished, finish_reason = self._is_sequence_finished(seq)
            if is_finished:
                self.scheduler.finish_sequence(seq)
            
            outputs.append(EngineOutput(
                seq_id=seq.seq_id,
                new_token_id=new_token_id,
                is_finished=is_finished,
                finish_reason=finish_reason,
                num_prompt_tokens=seq.prompt_len,
                num_generated_tokens=seq.num_tokens - seq.prompt_len
            ))

        return outputs

    def _is_sequence_finished(self, seq: Sequence) -> tuple[bool, FinishReason | None]:
        # Check for stop tokens
        if seq.token_ids[-1] == seq.sampling_params.eos_token_id and not seq.sampling_params.ignore_eos:
            return True, FinishReason.STOP
        
        # Check for max tokens
        if seq.sampling_params.max_tokens and seq.num_tokens >= seq.sampling_params.max_tokens:
            return True, FinishReason.LENGTH
        if seq.sampling_params.max_new_tokens and seq.num_tokens >= seq.prompt_len + seq.sampling_params.max_new_tokens:
            return True, FinishReason.LENGTH
        
        return False, None
    
    def wait_until_ready(self):
        self.model_executor.wait_until_ready()

    def shutdown(self):
        self.model_executor.shutdown()
        
    def has_unfinished_sequences(self):
        return self.scheduler.has_unfinished_sequences()
