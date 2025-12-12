import sys
import os
import signal
import psutil
import ctypes
import threading  
import functools
import traceback

def kill_itself_when_parent_died():
    """Set up process to be killed when parent dies (Linux only)."""
    if sys.platform == "linux":
        # sigkill this process when parent worker manager dies  
        PR_SET_PDEATHSIG = 1
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        except Exception as e:
            print(f"Warning: Failed to set PR_SET_PDEATHSIG: {e}")
    # On Windows and other platforms, this functionality is not available
    # but the process lifecycle is still managed through other mechanisms

def kill_process_tree(parent_pid = None, include_parent: bool = True, skip_pid: int | None = None):
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


def bind_parent_process_lifecycle(func):
    """函数装饰器实现当前进程和父进程生命周期绑定"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 当父进程死亡时，自动杀死当前进程
        kill_itself_when_parent_died()
        
        # 获取父进程
        parent_process = psutil.Process().parent()
        assert parent_process is not None, "Parent process not found."
        
        def _handle_exit(signum, frame):
            # 函数执行出错时，通知父进程并杀死自己
            try:
                if sys.platform == "win32":
                    # On Windows, SIGTERM might not be available, use terminate()
                    parent_process.terminate()
                else:
                    parent_process.send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # Parent already dead or no permission
            sys.exit(128 + signum if signum else 1)

        # 注册信号处理器
        # On Windows, only SIGINT and SIGBREAK are supported; SIGTERM is not available
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, _handle_exit)
        signal.signal(signal.SIGINT, _handle_exit)
        if sys.platform == "win32":
            try:
                signal.signal(signal.SIGBREAK, _handle_exit)  # type: ignore
            except AttributeError:
                pass  # SIGBREAK not available in this Python build
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            _handle_exit(signal.SIGTERM, None)

    return wrapper