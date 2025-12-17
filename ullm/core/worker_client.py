from typing import Optional
from .worker import Worker
import torch.multiprocessing as mp
from ..utils import bind_parent_process_lifecycle
import queue

class WorkerClient:
    def __init__(
        self,
        model: str,
        max_bs: int,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int = 29500,
        is_driver_worker = False,
        enforce_eager: bool = False,
        context_len: int = 2048,
    ):
        self.model = model
        self.max_bs = max_bs
        
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.rank = pp_rank * tp_size + tp_rank
        self.world_size = pp_size * tp_size

        self.nccl_port = nccl_port
        
        self.is_driver_worker = is_driver_worker
        self.enforce_eager = enforce_eager
        self.context_len = context_len        
        
        self.mp_ctx = mp.get_context('spawn')
        self.input_queue = self.mp_ctx.Queue()
        self.output_queue = self.mp_ctx.Queue()

        self.shutdown_event = self.mp_ctx.Event()
        self.init_worker()

    def init_worker(self):
        self.worker_process = self.mp_ctx.Process(
            target=self.worker_main_loop,
            name=f"worker-{self.rank}",
        )
        self.worker_process.start()

    def send_request(self, request_id: str, method: str, *args, **kwargs):
        self.input_queue.put_nowait((request_id, method, args, kwargs))

    def recv_response(self, timeout: Optional[float] = None):
        return self.output_queue.get(timeout=timeout)
    
    def shutdown(self):
        self.shutdown_event.set()
    
    def join(self):
        self.worker_process.join()

    @bind_parent_process_lifecycle
    def worker_main_loop(self):
        worker = Worker(
            model=self.model,
            max_bs=self.max_bs,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            enforce_eager=self.enforce_eager,
            context_len=self.context_len,
        )
        try:
            worker.init_environment()
            while not self.shutdown_event.is_set():
                try:
                    msg = self.input_queue.get(timeout=0.1)  # 等待输入
                except queue.Empty:
                    continue
                request_id, method_name, args, kwargs = msg
                response = self.handle_request(worker, method_name, *args, **kwargs)  # 处理请求
                if self.is_driver_worker:
                    self.output_queue.put_nowait((request_id, response))
        finally:
            worker.destroy_environment()
            print(f"Worker {self.rank} has shut down.")

    def handle_request(self, worker: Worker, method_name: str, *args, **kwargs):
        # 查找并调用注册的方法
        method = getattr(worker, method_name, None) if method_name else None
        method = method if callable(method) else None
        if method:
            # 执行方法
            result = method(*args, **kwargs)
            # 返回成功结果
            return ('success', result)
        else:
            # 方法不存在
            return ('failed', f"Method '{method_name}' not found")
