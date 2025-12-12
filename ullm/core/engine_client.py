from .engine import Engine, EngineOutput
from .common import SamplingParams
import torch.multiprocessing as mp
import threading
import queue
from ..utils import bind_parent_process_lifecycle
import os
import sys

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
        use_threading: bool = False,
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
        self.use_threading = use_threading
        
        if use_threading:
            # Use threading mode (for Windows or single-GPU setups)
            self.input_queue = queue.Queue()  # type: ignore
            self.output_queue = queue.Queue()  # type: ignore
            self.ready_event = threading.Event()  # type: ignore
        else:
            # Use multiprocessing mode (default)
            self.mp_ctx = mp.get_context('spawn')
            self.input_queue = self.mp_ctx.Queue()  # type: ignore
            self.output_queue = self.mp_ctx.Queue()  # type: ignore
            self.ready_event = self.mp_ctx.Event()  # type: ignore
        
        self.init_process()

    def init_process(self):
        if self.use_threading:
            self.engine_thread = threading.Thread(
                target=self.engine_main_loop,
                name="engine",
                daemon=False,
            )
            self.engine_thread.start()
        else:
            self.engine_process = self.mp_ctx.Process(
                target=self.engine_main_loop,
                name=f"engine",
            )
            self.engine_process.start()

    def engine_main_loop(self):
        # Only bind lifecycle for multiprocessing mode
        if not self.use_threading:
            bind_parent_process_lifecycle(self._engine_loop)()
        else:
            self._engine_loop()
    
    def _engine_loop(self):
        # Make child process session leader to avoid terminal signals (Linux only)
        if not self.use_threading and sys.platform != "win32" and hasattr(os, 'setsid'):
            os.setsid()
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
            use_threading=self.use_threading,
        )
        engine.wait_until_ready()
        self.ready_event.set()
        
        while True:
            is_shutdown = False
            # 如果 engine 中没有未完成的请求了，即 engine 的 waiting 和 running 队列都空了
            # 则在调用下一次 step 之前，阻塞等待一个新的请求
            if not engine.has_unfinished_sequences():
                method, *params = self.input_queue.get()
                if method == "shutdown":
                    is_shutdown = True
                elif method == "abort":
                    engine.abort_sequence(*params)
                elif method == "add":
                    engine.add_sequence(*params)
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            # 其他情况下，即存在未完成的请求，则需要继续调用 step
            while not self.input_queue.empty():
                method, *params = self.input_queue.get()
                if method == "shutdown":
                    is_shutdown = True
                    break
                elif method == "abort":
                    engine.abort_sequence(*params)
                elif method == "add":
                    engine.add_sequence(*params)
                else:
                    raise ValueError(f"Unknown method: {method}")
                    
            if is_shutdown:
                break
            
            outputs = engine.step()
            if outputs:
                if self.use_threading:
                    self.output_queue.put(outputs)
                else:
                    self.output_queue.put_nowait(outputs)

        if self.use_threading:
            self.output_queue.put(None)
        else:
            self.output_queue.put_nowait(None)
        engine.shutdown()
    
    def wait_until_ready(self):
        self.ready_event.wait()

    def shutdown(self):
        if self.use_threading:
            self.input_queue.put(("shutdown",))
            self.engine_thread.join()
        else:
            self.input_queue.put_nowait(("shutdown",))
            self.engine_process.join()
        print("Engine has been shut down.")

    def add_sequence(
        self,
        sequence_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams
    ):
        if self.use_threading:
            self.input_queue.put(("add", sequence_id, prompt_token_ids, sampling_params))
        else:
            self.input_queue.put_nowait(("add", sequence_id, prompt_token_ids, sampling_params))

    def get_output(self) -> list[EngineOutput]:
        return self.output_queue.get()

    def abort_sequence(self, sequence_id: str):
        if self.use_threading:
            self.input_queue.put(("abort", sequence_id))
        else:
            self.input_queue.put_nowait(("abort", sequence_id))
