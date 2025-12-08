from typing import Optional
from .worker import Worker
import torch.multiprocessing as mp
from ..utils import bind_parent_process_lifecycle

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
        self.methods = {}  # 用于存储注册的方法
        
        
        self.mp_ctx = mp.get_context('spawn')
        self.input_queue = self.mp_ctx.Queue()
        self.output_queue = self.mp_ctx.Queue()
        self.ready_event = self.mp_ctx.Event()
        self.init_worker()

    def init_worker(self):
        self.worker_process = self.mp_ctx.Process(
            target=self.worker_main_loop,
            name=f"worker-{self.rank}",
        )
        self.worker_process.start()

    def send_request(self, request_id: str, data: dict):
        self.input_queue.put_nowait((request_id, data))

    def recv_response(self, timeout: Optional[float] = None):
        return self.output_queue.get(timeout=timeout)
    
    def shutdown(self):
        self.input_queue.put_nowait('shutdown')
    
    def join(self):
        self.worker_process.join()
    
    def wait_until_ready(self, timeout: Optional[float] = None):
        self.ready_event.wait(timeout=timeout)

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
        worker.init_environment()
        worker.load_model()
        
        self.methods["execute_model"] = worker.execute_model
        self.methods["initialize_kv_cache"] = worker.initialize_kv_cache
        self.methods["profile_kv_cache_size"] = worker.profile_kv_cache_size
        
        self.ready_event.set()
        while True:
            msg = self.input_queue.get()  # 等待输入
            if msg == 'shutdown':
                break
            request_id, data = msg
            response = self.handle_request(data)  # 处理请求
            if self.is_driver_worker:
                self.output_queue.put_nowait((request_id, response))
        worker.destroy_environment()
        print(f"Worker {self.rank} has shut down.")

    def handle_request(self, request):
        """处理单个请求"""
        method_name = request['method']
        args = request.get('args', tuple())
        kwargs = request.get('kwargs', dict())
        # 查找并调用注册的方法
        if method_name in self.methods:
            method = self.methods[method_name]
            # 执行方法
            result = method(*args, **kwargs)
            
            # 返回成功结果
            return {
                'status': 'success',
                'result': result
            }
        else:
            # 方法不存在
            return {
                'status': 'error',
                'error': f"Method '{method_name}' not found"
            }
