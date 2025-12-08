import os
import uuid
import threading
from concurrent.futures import Future
from .worker_client import WorkerClient
from .common import ForwardBatch

class Executor:
    def __init__(
        self,
        model: str,
        max_bs: int,
        tp_size: int,
        pp_size: int,
        nccl_port: int = 29500,
        device_ids: list[int] | None = None,
        enforce_eager: bool = False,
        context_len: int = 2048,
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nccl_port = nccl_port
        self.device_ids = device_ids
        
        assert device_ids is None or len(device_ids) == tp_size * pp_size , \
            "device_ids should have the same length as tp_size * pp_size"
        if device_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

        self.workers: list[WorkerClient] = []
        self.driver_worker: WorkerClient
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                is_driver_worker = tp_rank == 0 and pp_rank == pp_size - 1
                worker = WorkerClient(
                    model=model,
                    max_bs=max_bs,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    nccl_port=self.nccl_port,
                    is_driver_worker=is_driver_worker,
                    enforce_eager=enforce_eager,
                    context_len=context_len,
                )
                if is_driver_worker:
                    self.driver_worker = worker
                self.workers.append(worker)

        self.pending: dict[str, Future[list[int]]] = {}  # 跟踪进行中的请求 {request_id: future}
        self.collect_thread = threading.Thread(
            target=self._collect_loop
        )
        self.collect_thread.start()

    def _collect_loop(self):
        while True:
            msg = self.driver_worker.recv_response()
            if msg == "shutdown":
                break
            request_id, data = msg
            future = self.pending.pop(request_id, None)
            if future:
                assert isinstance(data, dict)
                if data['status'] == 'success':
                    future.set_result(data['result'])
                else:
                    future.set_exception(Exception(data['error']))
        print("Executor stopped response collection.")

    def wait_until_ready(self):
        for worker in self.workers:
            worker.wait_until_ready()

    def shutdown(self):
        for worker in self.workers:
            worker.shutdown()
        for worker in self.workers:
            worker.join()
        self.driver_worker.output_queue.put_nowait("shutdown")
        self.collect_thread.join()

        print("Executor has been shut down.")

    def submit(self, method, *args, **kwargs):
        future = Future()
        request_id = uuid.uuid4().hex
        request = {
            'method': method,
            'args': args,
            'kwargs': kwargs
        }
        for worker in self.workers:
            worker.send_request(request_id, request)
        self.pending[request_id] = future
        return future

    def execute_model(self, batch: ForwardBatch) -> Future[list[int]]:
        return self.submit("execute_model", batch)
    
    def initialize_kv_cache(self, kv_cache_size: int):
        self.submit("initialize_kv_cache", kv_cache_size).result()

    def profile_kv_cache_size(self, gpu_memory_utilization: float) -> int:
        return self.submit("profile_kv_cache_size", gpu_memory_utilization).result()