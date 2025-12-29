import enum
import logging
import multiprocessing as mp
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import uuid
import weakref
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Callable

import zmq

from ..config import EngineConfig
from ..core.worker import Worker
from ..logger import init_logger
from ..utils import cleanup_resources
from .common import ForwardBatch

logger = init_logger(__name__)


class ResponseStatus(enum.Enum):
    SUCCESS = enum.auto()
    FAILED = enum.auto()


class WorkerEventType(enum.Enum):
    STARTUP = enum.auto()
    READY = enum.auto()
    SHUTDOWN = enum.auto()
    ERROR = enum.auto()
    DEAD = enum.auto()


@dataclass
class WorkerEvent:
    rank: int
    type: WorkerEventType


def handle_rpc_request(worker: Worker, method_name: str, *args, **kwargs):
    # 查找并调用注册的方法
    method = getattr(worker, method_name, None) if method_name else None
    method = method if callable(method) else None
    try:
        if method:
            result = method(*args, **kwargs)
            return ResponseStatus.SUCCESS, result
        else:
            # 方法不存在
            raise ValueError(f"Unknown method: {method_name}")
    except Exception as e:
        e.add_note(traceback.format_exc())
        return ResponseStatus.FAILED, e


def run_worker_loop(
    config: EngineConfig,
    rank: int,
    tp_rank: int,
    pp_rank: int,
    is_driver_worker: bool,
    report_pipe: Connection,
    broadcast_path: str,
    response_path: str,
):
    report_pipe.send(WorkerEvent(rank=rank, type=WorkerEventType.STARTUP))

    logging.basicConfig(level=config.log_level.upper())

    shutdown_requested = False

    def signal_handler(signum, frame):
        if not shutdown_requested:
            raise SystemExit()

    # Either SIGTERM or SIGINT will terminate the worker
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker: Worker | None = None
    heartbeat_thread: threading.Thread | None = None
    try:
        ctx = zmq.Context()

        input_queue = queue.Queue()
        output_queue: queue.Queue | None = None

        def process_input_socket():
            input_socket = ctx.socket(zmq.SUB)
            # Subscribe to all messages, if not set, all messages will be dropped
            input_socket.setsockopt(zmq.SUBSCRIBE, b"")
            input_socket.connect(broadcast_path)
            while True:
                frames = input_socket.recv_multipart(copy=False)
                msg = pickle.loads(frames[0].bytes)
                input_queue.put_nowait(msg)

        input_thread = threading.Thread(target=process_input_socket, daemon=True)
        input_thread.start()

        if is_driver_worker:
            output_queue = queue.Queue()

            def process_output_socket():
                output_socket = ctx.socket(zmq.PUSH)
                output_socket.connect(response_path)
                while True:
                    outputs = output_queue.get()
                    frames = [pickle.dumps(outputs)]
                    output_socket.send_multipart(frames, copy=False)

            output_thread = threading.Thread(target=process_output_socket, daemon=True)
            output_thread.start()

        worker = Worker(config=config, rank=rank, tp_rank=tp_rank, pp_rank=pp_rank)
        worker.init_environment()

        def heartbeat_loop():
            while not shutdown_requested:
                try:
                    report_pipe.send(WorkerEvent(rank=rank, type=WorkerEventType.READY))
                except (BrokenPipeError, OSError):
                    break  # 管道已关闭，退出心跳线程
                time.sleep(5)

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        while True:
            try:
                msg = input_queue.get(timeout=0.1)  # 等待输入
            except queue.Empty:
                continue
            request_id, method_name, args, kwargs = msg
            status, resp = handle_rpc_request(
                worker, method_name, *args, **kwargs
            )  # 处理请求
            if output_queue is not None:
                output_queue.put_nowait((request_id, status, resp))
    except SystemExit:
        shutdown_requested = True
        logger.debug(f"Worker {rank} process exit.")
        raise
    except Exception:
        shutdown_requested = True
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join()
        try:
            report_pipe.send(WorkerEvent(rank=rank, type=WorkerEventType.ERROR))
        except (BrokenPipeError, OSError):
            pass  # 管道已关闭是正常的
        if not worker:
            logger.exception(f"Worker {rank} initialization failed.")
        else:
            logger.exception(f"Worker {rank} encountered an error.")
    finally:
        shutdown_requested = True
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join()
        try:
            report_pipe.send(WorkerEvent(rank=rank, type=WorkerEventType.SHUTDOWN))
        except (BrokenPipeError, OSError):
            pass  # 管道已关闭是正常的
        if worker:
            worker.destroy_environment()


@dataclass
class WorkerProc:
    proc: BaseProcess
    rank: int
    is_driver_worker: bool
    report_pipe: Connection

    @staticmethod
    def create(
        config: EngineConfig,
        rank: int,
        tp_rank: int,
        pp_rank: int,
        is_driver_worker: bool,
        broadcast_path: str,
        response_path: str,
    ) -> "WorkerProc":
        mp_ctx = mp.get_context("spawn")

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
                broadcast_path,
                response_path,
            ),
        )
        proc.start()

        return WorkerProc(
            proc=proc,
            rank=rank,
            is_driver_worker=is_driver_worker,
            report_pipe=report_reader,
        )


class MultiWorkerClient:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.workers: list[WorkerProc] = []
        self.driver_worker: WorkerProc

        self.mp_ctx = mp.get_context("spawn")
        self.zmq_ctx = zmq.Context()

        self.broadcast_socket = self.zmq_ctx.socket(zmq.PUB)
        self.broadcast_path = "ipc:///tmp/ullm_executor_broadcast"
        self.broadcast_socket.bind(self.broadcast_path)

        self.response_socket = self.zmq_ctx.socket(zmq.PULL)
        self.response_path = "ipc:///tmp/ullm_executor_response"
        self.response_socket.bind(self.response_path)

        self.callbacks: dict[WorkerEventType, list[Callable[[WorkerEvent], None]]] = {
            type: [] for type in WorkerEventType
        }
        self.callbacks_lock = threading.Lock()

        self.is_dead = False

        self.set_envs()
        self.start_workers()
        self.start_worker_monitor()
        self.wait_worker_ready()

    def add_worker_event_listener(
        self, event_type: WorkerEventType, callback: Callable[[WorkerEvent], None]
    ):
        if event_type in self.callbacks:
            self.callbacks_lock.acquire()
            self.callbacks[event_type].append(callback)
            self.callbacks_lock.release()

    def remove_worker_event_listener(
        self, event_type: WorkerEventType, callback: Callable[[WorkerEvent], None]
    ):
        if event_type in self.callbacks:
            self.callbacks_lock.acquire()
            self.callbacks[event_type].remove(callback)
            self.callbacks_lock.release()

    def on_failure(self, callback: Callable[[], None]):
        self.add_worker_event_listener(
            WorkerEventType.DEAD,
            lambda event: callback(),
        )

    def shutdown(self):
        logger.debug(f"Shutting down {self.__class__.__name__}...")
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
                rank = pp_rank * self.tp_size + tp_rank
                is_driver_worker = tp_rank == 0 and pp_rank == self.pp_size - 1
                logger.debug(
                    f"Starting worker process for TP rank {tp_rank}, PP rank {pp_rank} (driver: {is_driver_worker})..."
                )
                worker = WorkerProc.create(
                    config=self.config,
                    rank=rank,
                    tp_rank=tp_rank,
                    pp_rank=pp_rank,
                    is_driver_worker=is_driver_worker,
                    broadcast_path=self.broadcast_path,
                    response_path=self.response_path,
                )
                if is_driver_worker:
                    self.driver_worker = worker
                self.workers.append(worker)
        self._finalizer = weakref.finalize(
            self,
            cleanup_resources,
            processes=[w.proc for w in self.workers],
            sockets=[self.broadcast_socket, self.response_socket],
        )

    def wait_worker_ready(self):
        logger.debug("Waiting for all workers to be ready...")
        unready = [w.rank for w in self.workers]
        q: queue.Queue[WorkerEvent] = queue.Queue()
        dead_event = threading.Event()

        def ready_callback(event: WorkerEvent):
            q.put(event)

        def dead_callback(event: WorkerEvent):
            dead_event.set()

        self.add_worker_event_listener(WorkerEventType.READY, ready_callback)
        self.add_worker_event_listener(WorkerEventType.DEAD, dead_callback)

        while unready:
            if dead_event.is_set():
                raise RuntimeError(
                    "Worker process encountered an error during startup."
                )
            try:
                event = q.get(timeout=1.0)
            except queue.Empty:
                continue
            if event.rank in unready:
                unready.remove(event.rank)

        self.remove_worker_event_listener(WorkerEventType.READY, ready_callback)
        self.remove_worker_event_listener(WorkerEventType.DEAD, dead_callback)
        logger.debug("All workers are ready.")

    def start_worker_monitor(self):
        workers = self.workers
        weak_self = weakref.ref(self)

        def worker_dead_monitor():
            sentinels = [w.proc.sentinel for w in workers]
            died = wait(sentinels)
            dead_worker = next(w for w in workers if w.proc.sentinel == died[0])
            self_ref = weak_self()
            if not self_ref:
                return
            logger.debug(
                f"Worker {dead_worker.rank} event triggered: {WorkerEventType.DEAD}."
            )
            self_ref.callbacks_lock.acquire()
            callbacks = list(self_ref.callbacks[WorkerEventType.DEAD])
            self_ref.callbacks_lock.release()
            for callback in callbacks:
                callback(
                    WorkerEvent(
                        rank=dead_worker.rank,
                        type=WorkerEventType.DEAD,
                    )
                )
            self_ref.is_dead = True
            self_ref.shutdown()

        def worker_event_monitor():
            conns = [w.report_pipe for w in workers]
            while True:
                for conn in wait(conns, timeout=1.0):
                    assert isinstance(conn, Connection)
                    try:
                        event = conn.recv()
                        assert isinstance(event, WorkerEvent)
                    except (EOFError, OSError):
                        continue
                    self_ref = weak_self()
                    if not self_ref:
                        return
                    logger.debug(f"Worker {event.rank} event triggered: {event.type}.")
                    self_ref.callbacks_lock.acquire()
                    callbacks = list(self_ref.callbacks[event.type])
                    self_ref.callbacks_lock.release()
                    for callback in callbacks:
                        callback(event)

        logger.debug("Starting worker monitor thread...")
        self.dead_monitor_thread = threading.Thread(
            target=worker_dead_monitor,
            name="worker-dead-monitor",
            daemon=True,
        )
        self.dead_monitor_thread.start()
        self.event_monitor_thread = threading.Thread(
            target=worker_event_monitor,
            name="worker-event-monitor",
            daemon=True,
        )
        self.event_monitor_thread.start()
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
        self.response_socket.setsockopt(zmq.RCVTIMEO, 1000)
        while not self.is_dead:
            try:
                frames = self.response_socket.recv_multipart(
                    flags=zmq.NOBLOCK, copy=False
                )
                msg = pickle.loads(frames[0].bytes)
            except zmq.Again:
                continue
            request_id, status, resp = msg
            future = self.pending.pop(request_id, None)
            if future:
                if status == ResponseStatus.SUCCESS:
                    future.set_result(resp)
                elif status == ResponseStatus.FAILED:
                    future.set_exception(resp)

    def submit(self, method, *args, **kwargs):
        if self.is_dead:
            raise RuntimeError("Executor has died.")
        future = Future()
        request_id = uuid.uuid4().hex
        frames = [pickle.dumps((request_id, method, args, kwargs))]
        self.broadcast_socket.send_multipart(frames, copy=False)
        self.pending[request_id] = future
        return future

    def execute_model(self, batch: ForwardBatch) -> Future[list[int]]:
        return self.submit("execute_model", batch)

    def initialize(self):
        self.submit("initialize", self.config.gpu_memory_utilization).result()

    def get_kv_cache_size(self) -> int:
        return self.submit("get_kv_cache_size").result()
