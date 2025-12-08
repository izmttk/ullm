from .engine import Engine, EngineOutput
from .common import SamplingParams
import torch.multiprocessing as mp
from ..utils import bind_parent_process_lifecycle
import os

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
        
        self.init_process()

    def init_process(self):
        self.engine_process = self.mp_ctx.Process(
            target=self.engine_main_loop,
            name=f"engine",
        )
        self.engine_process.start()

    @bind_parent_process_lifecycle
    def engine_main_loop(self):
        os.setsid()  # 使子进程成为新会话的首进程，防止收到来自终端的信号
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
                self.output_queue.put_nowait(outputs)

        self.output_queue.put_nowait(None)
        engine.shutdown()
    
    def wait_until_ready(self):
        self.ready_event.wait()

    def shutdown(self):
        self.input_queue.put_nowait(("shutdown",))
        self.engine_process.join()
        print("Engine has been shut down.")

    def add_sequence(
        self,
        sequence_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams
    ):
        self.input_queue.put_nowait(
            ("add", sequence_id, prompt_token_ids, sampling_params)
        )

    def get_output(self) -> list[EngineOutput]:
        return self.output_queue.get()

    def abort_sequence(self, sequence_id: str):
        self.input_queue.put_nowait(("abort", sequence_id))
