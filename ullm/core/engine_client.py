import enum
import logging
import queue
import signal
import threading
import time
import weakref
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait

import msgspec
import torch.multiprocessing as mp
import zmq

from ..config import EngineConfig
from ..logger import init_logger
from ..serial import MsgpackDecoder, MsgpackEncoder
from ..utils import cleanup_resources
from .common import EngineDeadError, EngineStepResult, SamplingParams
from .engine import Engine

logger = init_logger(__name__)


class EngineEventType(enum.Enum):
    STARTUP = enum.auto()
    READY = enum.auto()
    SHUTDOWN = enum.auto()
    ERROR = enum.auto()
    DEAD = enum.auto()


@dataclass
class EngineEvent:
    type: EngineEventType


class EngineRequestBase(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
    tag=True,
):
    sequence_id: str


class EngineRequestAdd(EngineRequestBase):
    prompt_token_ids: list[int]
    sampling_params: SamplingParams


class EngineRequestAbort(EngineRequestBase):
    pass


EngineRequest = EngineRequestAdd | EngineRequestAbort


class EngineOutputs(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    outputs: list[EngineStepResult]


def handle_engine_request(engine: Engine, req: EngineRequestBase):
    if isinstance(req, EngineRequestAdd):
        engine.add_sequence(
            sequence_id=req.sequence_id,
            prompt_token_ids=req.prompt_token_ids,
            sampling_params=req.sampling_params,
        )
    elif isinstance(req, EngineRequestAbort):
        engine.abort_sequence(sequence_id=req.sequence_id)
    else:
        raise ValueError(f"Invalid request: {type(req).__name__}")


def run_engine_loop(
    config: EngineConfig,
    report_pipe: Connection,
    input_path: str,
    output_path: str,
):
    report_pipe.send(EngineEvent(EngineEventType.STARTUP))
    logging.basicConfig(level=config.log_level.upper())

    shutdown_requested = False

    def signal_handler(signum, frame):
        if not shutdown_requested:
            raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    engine: Engine | None = None
    heartbeat_thread: threading.Thread | None = None
    engine_failed = False
    try:
        ctx = zmq.Context()

        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder(EngineRequest)

        input_queue: queue.Queue[EngineRequest] = queue.Queue()
        output_queue: queue.Queue[EngineOutputs] = queue.Queue()

        def process_input_sockets():
            input_socket = ctx.socket(zmq.PULL)
            input_socket.connect(input_path)
            while True:
                frames = input_socket.recv_multipart(copy=False)
                msg = decoder.decode(frames)
                assert isinstance(msg, EngineRequest)
                input_queue.put_nowait(msg)

        def process_output_sockets():
            output_socket = ctx.socket(zmq.PUSH)
            output_socket.connect(output_path)
            while True:
                outputs = output_queue.get()
                frames = encoder.encode(outputs)
                output_socket.send_multipart(frames, copy=False)

        input_thread = threading.Thread(target=process_input_sockets, daemon=True)
        output_thread = threading.Thread(target=process_output_sockets, daemon=True)
        input_thread.start()
        output_thread.start()

        def failure_callback():
            nonlocal engine_failed
            engine_failed = True

        engine = Engine(config=config, failure_callback=failure_callback)

        def heartbeat_loop():
            while not shutdown_requested:
                try:
                    report_pipe.send(EngineEvent(EngineEventType.READY))
                except (BrokenPipeError, OSError):
                    break  # 管道已关闭，退出心跳线程
                time.sleep(5)

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        while True:
            if engine_failed:
                raise RuntimeError("Engine has encountered a fatal error.")
            # 如果 engine 中没有未完成的请求了，即 engine 的 waiting 和 running 队列都空了
            # 则在调用下一次 step 之前，阻塞等待一个新的请求
            if not engine.has_unfinished_sequences():
                try:
                    req = input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                handle_engine_request(engine, req)

            # 其他情况下，即存在未完成的请求，则需要继续调用 step
            while not input_queue.empty():
                req = input_queue.get_nowait()
                handle_engine_request(engine, req)

            outputs = engine.step()
            if outputs:
                output_queue.put_nowait(EngineOutputs(outputs=outputs))
    except SystemExit:
        shutdown_requested = True
        logger.debug("Engine process exit.")
        raise
    except Exception:
        shutdown_requested = True
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join()
        try:
            report_pipe.send(EngineEvent(EngineEventType.ERROR))
        except (BrokenPipeError, OSError):
            pass  # 管道已关闭是正常的
        if not engine:
            logger.exception("Engine initialization failed.")
        else:
            logger.exception("Engine encountered an error.")
    finally:
        shutdown_requested = True
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join()
        try:
            report_pipe.send(EngineEvent(EngineEventType.SHUTDOWN))
        except (BrokenPipeError, OSError):
            pass  # 管道已关闭是正常的
        if engine:
            engine.shutdown()
            logger.debug("Engine has been shut down.")


class MpClient:
    def __init__(self, config: EngineConfig):
        logger.setLevel(config.log_level.upper())

        self.config = config
        self.mp_ctx = mp.get_context("spawn")

        self.zmq_ctx = zmq.Context()
        self.input_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.output_socket = self.zmq_ctx.socket(zmq.PULL)

        self.input_path = "ipc:///tmp/ullm_engine_input"
        self.output_path = "ipc:///tmp/ullm_engine_output"

        self.input_socket.bind(self.input_path)
        self.output_socket.bind(self.output_path)

        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineOutputs)

        self.is_dead = False

        self.start_engine()
        self.start_engine_monitor()
        self.wait_engine_ready()

    def shutdown(self):
        logger.debug(f"Shutting down {self.__class__.__name__}...")
        self._finalizer()
        logger.debug(f"{self.__class__.__name__} shut down.")

    def start_engine(self):
        logger.debug("Starting engine process...")
        report_reader, report_writer = self.mp_ctx.Pipe(duplex=False)
        self.report_pipe = report_reader
        self.engine_process = self.mp_ctx.Process(
            target=run_engine_loop,
            name="engine",
            args=(
                self.config,
                report_writer,
                self.input_path,
                self.output_path,
            ),
        )
        self.engine_process.start()
        self._finalizer = weakref.finalize(
            self,
            cleanup_resources,
            processes=[self.engine_process],
            sockets=[self.input_socket, self.output_socket],
        )

        logger.debug("Engine process started.")

    def wait_engine_ready(self):
        logger.debug("Waiting for engine to be ready...")
        while True:
            event: EngineEvent = self.report_pipe.recv()
            if event.type == EngineEventType.STARTUP:
                continue
            elif event.type == EngineEventType.READY:
                break
            else:
                raise RuntimeError(
                    "Engine process encountered an error during startup."
                )
        logger.debug("Engine process is ready.")

    def start_engine_monitor(self):
        if not hasattr(self, "engine_process") or not self.engine_process:
            raise RuntimeError("Engine process has not been started.")
        # Avoid circular references
        weak_self = weakref.ref(self)

        def engine_dead_monitor():
            _died = wait([self.engine_process.sentinel])
            self_ref = weak_self()
            if not self_ref:
                return
            logger.debug("Engine process has died.")
            self_ref.is_dead = True
            self_ref.shutdown()

        logger.debug("Starting engine monitor thread...")
        self.dead_monitor_thread = threading.Thread(
            target=engine_dead_monitor,
            name="engine-dead-monitor",
            daemon=True,
        )
        self.dead_monitor_thread.start()
        logger.debug("Engine monitor thread started.")


class EngineClient(MpClient):
    def add_sequence(
        self,
        sequence_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ):
        if self.is_dead:
            raise EngineDeadError()
        frames = self.encoder.encode(
            EngineRequestAdd(
                sequence_id=sequence_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
        )

        self.input_socket.send_multipart(
            frames,
            copy=False,
        )

    def abort_sequence(self, sequence_id: str):
        if self.is_dead:
            raise EngineDeadError()
        frames = self.encoder.encode(EngineRequestAbort(sequence_id=sequence_id))
        self.input_socket.send_multipart(frames, copy=False)

    def get_output(self) -> list[EngineStepResult]:
        self.output_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        while True:
            # raise an exception if engine process has died
            # this will help server to shutdown automatically
            if self.is_dead:
                raise EngineDeadError()
            try:
                frames = self.output_socket.recv_multipart(
                    flags=zmq.NOBLOCK, copy=False
                )
                outputs = self.decoder.decode(frames)
                assert isinstance(outputs, EngineOutputs)
                break
            except zmq.Again:
                continue
        return outputs.outputs
