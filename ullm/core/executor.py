import enum
import multiprocessing as mp
import os
import queue
import signal
import threading
import uuid
import weakref
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess

from ..config import EngineConfig
from ..core.worker import Worker
from ..logger import init_logger
from ..utils import shutdown
from .common import ForwardBatch

logger = init_logger(__name__)


class ResponseStatus(enum.Enum):
    SUCCESS = 0
    ERROR = 1


def run_worker_loop(
    config: EngineConfig,
    rank: int,
    tp_rank: int,
    pp_rank: int,
    is_driver_worker: bool,
    report_pipe: Connection,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    report_pipe.send(b"HELLO")
    logger.setLevel(config.log_level.upper())

    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    # Either SIGTERM or SIGINT will terminate the worker
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker = None
    try:
        worker = Worker(config=config, rank=rank, tp_rank=tp_rank, pp_rank=pp_rank)
        worker.init_environment()

        report_pipe.send(b"READY")
        while True:
            try:
                msg = input_queue.get(timeout=0.1)  # 等待输入
            except queue.Empty:
                continue
            request_id, method_name, args, kwargs = msg
            status, resp = handle_rpc_request(
                worker, method_name, *args, **kwargs
            )  # 处理请求
            if is_driver_worker:
                output_queue.put_nowait((request_id, status, resp))
    except SystemExit:
        report_pipe.send(b"EXIT")
        logger.debug(f"Worker {rank} process exit.")
        raise
    except Exception:
        report_pipe.send(b"ERROR")
        if not worker:
            logger.exception(f"Worker {rank} initialization failed.")
        else:
            logger.exception(f"Worker {rank} encountered an error.")
        shutdown_requested = True
        raise
    finally:
        if worker:
            worker.destroy_environment()


def handle_rpc_request(worker: Worker, method_name: str, *args, **kwargs):
    # 查找并调用注册的方法
    method = getattr(worker, method_name, None) if method_name else None
    method = method if callable(method) else None
    if method:
        try:
            result = method(*args, **kwargs)
            return ResponseStatus.SUCCESS, result
        except Exception as e:
            return ResponseStatus.ERROR, str(e)
    else:
        # 方法不存在
        return ResponseStatus.ERROR, f"Unknown method: {method_name}"


@dataclass
class WorkerProc:
    proc: BaseProcess
    rank: int
    report_pipe: Connection
    input_queue: mp.Queue
    output_queue: mp.Queue

    @staticmethod
    def create(
        config: EngineConfig,
        rank: int,
        tp_rank: int,
        pp_rank: int,
        is_driver_worker: bool,
    ) -> "WorkerProc":
        mp_ctx = mp.get_context("spawn")
        input_queue = mp_ctx.Queue()
        output_queue = mp_ctx.Queue()
        report_reader, report_writer = mp.Pipe(duplex=False)

        proc = mp_ctx.Process(
            target=run_worker_loop,
            name=f"worker-{rank}",
            args=(
                config,
                rank,
                tp_rank,
                pp_rank,
                is_driver_worker,
                report_writer,
                input_queue,
                output_queue,
            ),
        )
        proc.start()

        return WorkerProc(
            proc=proc,
            rank=rank,
            report_pipe=report_reader,
            input_queue=input_queue,
            output_queue=output_queue,
        )


class MultiWorkerClient:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.workers: list[WorkerProc] = []
        self.driver_worker: WorkerProc

        self.mp_ctx = mp.get_context("spawn")
        self.shutting_down = False

        self.set_envs()
        self.start_workers()
        self.start_worker_monitor()
        self.wait_worker_ready()

    def shutdown(self):
        logger.debug(f"Shutting down {self.__class__.__name__}...")
        self.shutting_down = True
        self._finalizer()
        logger.debug(f"{self.__class__.__name__} shut down.")

    def set_envs(self):
        assert (
            self.config.device_ids is None
            or len(self.config.device_ids) == self.tp_size * self.pp_size
        ), "device_ids should have the same length as tp_size * pp_size"
        if self.config.device_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, self.config.device_ids)
            )

    def start_workers(self):
        for pp_rank in range(self.pp_size):
            for tp_rank in range(self.tp_size):
                is_driver_worker = tp_rank == 0 and pp_rank == self.pp_size - 1
                logger.debug(
                    f"Starting worker process for TP rank {tp_rank}, PP rank {pp_rank} (driver: {is_driver_worker})..."
                )
                worker = WorkerProc.create(
                    config=self.config,
                    rank=pp_rank * self.tp_size + tp_rank,
                    tp_rank=tp_rank,
                    pp_rank=pp_rank,
                    is_driver_worker=is_driver_worker,
                )
                if is_driver_worker:
                    self.driver_worker = worker
                self.workers.append(worker)
        self._finalizer = weakref.finalize(
            self, shutdown, [w.proc for w in self.workers]
        )

    def wait_worker_ready(self):
        logger.debug("Waiting for all workers to be ready...")
        unready = [w.report_pipe for w in self.workers]
        while not self.shutting_down and unready:
            for conn in wait(unready):
                assert isinstance(conn, Connection)
                msg = conn.recv()
                if msg == b"HELLO":
                    continue
                elif msg == b"READY":
                    unready.remove(conn)
                    continue
                elif msg == b"EXIT":
                    unready.remove(conn)
                    continue
                elif msg == b"ERROR":
                    raise RuntimeError(
                        "Worker process encountered an error during startup."
                    )
        logger.debug("All workers are ready.")

    def start_worker_monitor(self):
        workers = self.workers
        weak_self = weakref.ref(self)

        def worker_monitor():
            sentinels = [w.proc.sentinel for w in workers]
            died = wait(sentinels)
            dead_worker = next(w for w in workers if w.proc.sentinel == died[0])
            self_ref = weak_self()
            if not self_ref or self_ref.shutting_down:
                return
            logger.error(
                f"Worker process {dead_worker.proc.name} has exited unexpectedly."
            )
            self_ref.shutdown()

        logger.debug("Starting worker monitor thread...")
        self.monitor_thread = threading.Thread(
            target=worker_monitor,
            name="worker-monitor",
            daemon=True,
        )
        self.monitor_thread.start()
        logger.debug("Worker monitor thread started.")


class Executor(MultiWorkerClient):
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.pending: dict[
            str, Future[list[int]]
        ] = {}  # 跟踪进行中的请求 {request_id: future}
        self.collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.collect_thread.start()

    def _collect_loop(self):
        while True:
            try:
                msg = self.driver_worker.output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            request_id, status, resp = msg
            future = self.pending.pop(request_id, None)
            if future:
                if status == ResponseStatus.SUCCESS:
                    future.set_result(resp)
                elif status == ResponseStatus.ERROR:
                    future.set_exception(Exception(resp))

    def submit(self, method, *args, **kwargs):
        future = Future()
        request_id = uuid.uuid4().hex
        for worker in self.workers:
            worker.input_queue.put_nowait((request_id, method, args, kwargs))
        self.pending[request_id] = future
        return future

    def execute_model(self, batch: ForwardBatch) -> Future[list[int]]:
        return self.submit("execute_model", batch)

    def initialize(self):
        self.submit("initialize", self.config.gpu_memory_utilization).result()

    def get_kv_cache_size(self) -> int:
        return self.submit("get_kv_cache_size").result()
