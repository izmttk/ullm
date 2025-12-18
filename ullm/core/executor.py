import os
import uuid
import threading
from concurrent.futures import Future
from .worker_client import WorkerClient
from .common import ForwardBatch
import queue

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
            target=self._collect_loop,
            daemon=True
        )
        self.collect_thread.start()

    def _collect_loop(self):
        while True:
            try:
                msg = self.driver_worker.recv_response(timeout=0.1)
            except queue.Empty:
                continue
            request_id, resp = msg
            future = self.pending.pop(request_id, None)
            if future:
                future.set_result(resp)

    def shutdown(self):
        for worker in self.workers:
            worker.shutdown()
        for worker in self.workers:
            worker.join()
        print("Executor has been shut down.")

    def submit(self, method, *args, **kwargs):
        future = Future()
        request_id = uuid.uuid4().hex
        for worker in self.workers:
            worker.send_request(request_id, method, *args, **kwargs)
        self.pending[request_id] = future
        return future

    def execute_model(self, batch: ForwardBatch) -> Future[list[int]]:
        return self.submit("execute_model", batch)

    def initialize(self, gpu_memory_utilization: float):
        self.submit("initialize", gpu_memory_utilization).result()

    def get_kv_cache_size(self) -> int:
        return self.submit("get_kv_cache_size").result()
