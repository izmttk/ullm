from .engine import Engine, EngineOutput
from .common import SamplingParams
import torch.multiprocessing as mp
from ..utils import bind_parent_process_lifecycle, kill_process_tree
import os
import queue

class EngineClient:
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
        self.model = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_bs = max_bs
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nccl_port = nccl_port
        self.device_ids = device_ids
        self.enforce_eager = enforce_eager
        self.context_len = context_len
        
        self.mp_ctx = mp.get_context('spawn')
        self.input_queue = self.mp_ctx.Queue()
        self.output_queue = self.mp_ctx.Queue()
        
        self.ready_event = self.mp_ctx.Event()
        self.shutdown_event = self.mp_ctx.Event()
        
        self.init_process()

    def init_process(self):
        self.engine_process = self.mp_ctx.Process(
            target=self.engine_main_loop,
            name=f"engine",
        )
        self.engine_process.start()

    @bind_parent_process_lifecycle
    def engine_main_loop(self):
        engine = None
        try:
            engine = Engine(
                model=self.model,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_bs=self.max_bs,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                nccl_port=self.nccl_port,
                device_ids=self.device_ids,
                enforce_eager=self.enforce_eager,
                context_len=self.context_len,
            )
            self.ready_event.set()
            while not self.shutdown_event.is_set():
                # 如果 engine 中没有未完成的请求了，即 engine 的 waiting 和 running 队列都空了
                # 则在调用下一次 step 之前，阻塞等待一个新的请求
                if not engine.has_unfinished_sequences():
                    try:
                        method_name, args, kwargs = self.input_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    method = getattr(engine, method_name, None) if method_name else None
                    method = method if callable(method) else None
                    if method:
                        method(*args, **kwargs)
                    else:
                        raise ValueError(f"Unknown method: {method_name}")
                
                # 其他情况下，即存在未完成的请求，则需要继续调用 step
                while not self.input_queue.empty() and not self.shutdown_event.is_set():
                    method_name, args, kwargs = self.input_queue.get_nowait()
                    method = getattr(engine, method_name, None) if method_name else None
                    method = method if callable(method) else None
                    if method:
                        method(*args, **kwargs)
                    else:
                        raise ValueError(f"Unknown method: {method_name}")
                    
                if self.shutdown_event.is_set():
                    break

                outputs = engine.step()
                if outputs:
                    self.output_queue.put_nowait(outputs)
        finally:
            self.shutdown_event.set() # 在非正常退出时，无条件设置 shutdown_event
            self.ready_event.set() # 同上
            if engine:
                engine.shutdown()
            print("Engine has been shut down.")

    def wait_until_ready(self):
        self.ready_event.wait()
        if self.shutdown_event.is_set():
            raise RuntimeError("Engine initialization failed.")

    def shutdown(self):
        self.shutdown_event.set()
        self.engine_process.join(timeout=5)
        if self.engine_process.is_alive():
            kill_process_tree(self.engine_process.pid)

    def add_sequence(
        self,
        sequence_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams
    ):
        self.input_queue.put_nowait(
            ("add_sequence", (sequence_id, prompt_token_ids, sampling_params), {})
        )

    def get_output(self, timeout: float | None) -> list[EngineOutput]:
        return self.output_queue.get(timeout=timeout)

    def abort_sequence(self, sequence_id: str):
        self.input_queue.put_nowait(
            ("abort_sequence", (sequence_id,), {})
        )
