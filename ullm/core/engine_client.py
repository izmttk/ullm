import queue
import signal
import threading
import weakref
from multiprocessing.connection import Connection, wait

import torch.multiprocessing as mp

from ..config import EngineConfig
from ..logger import init_logger
from ..utils import shutdown
from .common import EngineOutput, SamplingParams
from .engine import Engine

logger = init_logger(__name__)


def run_engine_loop(
    config: EngineConfig,
    report_pipe: Connection,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    report_pipe.send(b"HELLO")
    logger.setLevel(config.log_level)

    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    engine: Engine | None = None
    try:
        engine = Engine(config=config)
        report_pipe.send(b"READY")
        while True:
            # 如果 engine 中没有未完成的请求了，即 engine 的 waiting 和 running 队列都空了
            # 则在调用下一次 step 之前，阻塞等待一个新的请求
            if not engine.has_unfinished_sequences():
                try:
                    method_name, args, kwargs = input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                method = getattr(engine, method_name, None) if method_name else None
                method = method if callable(method) else None
                if method:
                    method(*args, **kwargs)
                else:
                    raise ValueError(f"Unknown method: {method_name}")

            # 其他情况下，即存在未完成的请求，则需要继续调用 step
            while not input_queue.empty():
                method_name, args, kwargs = input_queue.get_nowait()
                method = getattr(engine, method_name, None) if method_name else None
                method = method if callable(method) else None
                if method:
                    method(*args, **kwargs)
                else:
                    raise ValueError(f"Unknown method: {method_name}")

            outputs = engine.step()
            if outputs:
                output_queue.put_nowait(outputs)
    except SystemExit:
        report_pipe.send(b"EXIT")
        logger.debug("Engine process exit.")
        raise
    except Exception:
        report_pipe.send(b"ERROR")
        if not engine:
            logger.exception("Engine initialization failed.")
        else:
            logger.exception("Engine encountered an error.")
        raise
    finally:
        if engine:
            engine.shutdown()
            logger.debug("Engine has been shut down.")


class MpClient:
    def __init__(self, config: EngineConfig):
        logger.setLevel(config.log_level)

        self.config = config
        self.mp_ctx = mp.get_context("spawn")
        self.input_queue = self.mp_ctx.Queue()
        self.output_queue = self.mp_ctx.Queue()
        self.shutting_down = False

        weak_self = weakref.ref(self)

        def cleanup():
            self_ref = weak_self()
            if self_ref and hasattr(self_ref, "input_queue"):
                self_ref.input_queue.close()
            if self_ref and hasattr(self_ref, "output_queue"):
                self_ref.output_queue.close()
            if self_ref and hasattr(self_ref, "engine_process"):
                shutdown(self_ref.engine_process)

        self._finalizer = weakref.finalize(self, cleanup)
        self.start_engine()

    def shutdown(self):
        logger.debug(f"Shutting down {self.__class__.__name__}...")
        self.shutting_down = True
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
                self.input_queue,
                self.output_queue,
            ),
        )
        self.engine_process.start()
        logger.debug("Engine process started.")

        self.start_engine_monitor()
        self.wait_engine_ready()

    def wait_engine_ready(self):
        logger.debug("Waiting for engine to be ready...")
        while not self.shutting_down:
            for conn in wait([self.report_pipe], timeout=1.0):
                assert isinstance(conn, Connection)
                msg = conn.recv()
                if msg == b"HELLO":
                    continue
                elif msg == b"READY":
                    logger.debug("Engine process is ready.")
                    return
                elif msg == b"EXIT":
                    raise RuntimeError("Engine process exited during startup.")
                elif msg == b"ERROR":
                    raise RuntimeError(
                        "Engine process encountered an error during startup."
                    )

    def start_engine_monitor(self):
        if not hasattr(self, "engine_process") or not self.engine_process:
            raise RuntimeError("Engine process has not been started.")
        # Avoid circular references
        weak_self = weakref.ref(self)

        def run_monitor():
            _died = wait([self.engine_process.sentinel])
            self_ref = weak_self()
            if not self_ref or self_ref.shutting_down:
                return
            logger.error("Engine process has exited unexpectedly.")
            self_ref.shutdown()

        logger.debug("Starting engine monitor thread...")
        self.monitor_thread = threading.Thread(
            target=run_monitor,
            name="engine-monitor",
            daemon=True,
        )
        self.monitor_thread.start()
        logger.debug("Engine monitor thread started.")


class EngineClient(MpClient):
    def add_sequence(
        self,
        sequence_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ):
        self.input_queue.put_nowait(
            ("add_sequence", (sequence_id, prompt_token_ids, sampling_params), {})
        )

    def abort_sequence(self, sequence_id: str):
        self.input_queue.put_nowait(("abort_sequence", (sequence_id,), {}))

    def get_output(self, timeout: float | None) -> list[EngineOutput]:
        return self.output_queue.get(timeout=timeout)
