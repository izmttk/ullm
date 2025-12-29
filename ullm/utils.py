import os
import signal
import sys
import threading
import time
from multiprocessing.process import BaseProcess
from typing import Sequence

import psutil
import zmq
import zmq.asyncio


def kill_process_tree(
    parent_pid=None, include_parent: bool = True, skip_pid: int | None = None
):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


# shutdown function cannot be a bound method,
# else the gc cannot collect the object.
def shutdown(procs: list[BaseProcess] | BaseProcess):
    proc_list = procs if isinstance(procs, list) else [procs]
    # Shutdown the process.
    for proc in proc_list:
        if proc.is_alive():
            proc.terminate()

    # Allow 5 seconds for remaining procs to terminate.
    deadline = time.monotonic() + 5
    for proc in proc_list:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    for proc in proc_list:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)


def close_sockets(sockets: Sequence[zmq.Socket | zmq.asyncio.Socket]):
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


def cleanup_resources(
    processes: list[BaseProcess] | BaseProcess | None = None,
    sockets: list[zmq.Socket] | None = None,
):
    if processes:
        shutdown(processes)
    # ZMQ context termination can hang if the sockets
    # aren't explicitly closed first.
    if sockets:
        close_sockets(sockets)
