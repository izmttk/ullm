import logging
import sys
import typing
from functools import lru_cache, partial


class ColoredFormatter(logging.Formatter):
    """Adds ANSI color codes to log levels for terminal output.

    This formatter adds colors by injecting them into the format string for
    static elements (timestamp, filename, line number) and modifying the
    levelname attribute for dynamic color selection.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    GREY = "\033[90m"  # Grey for timestamp and file info
    RESET = "\033[0m"

    def __init__(self, fmt, datefmt=None, style="%"):
        # Inject grey color codes into format string for timestamp and file info
        if fmt:
            # Wrap %(asctime)s with grey
            fmt = fmt.replace("%(asctime)s", f"{self.GREY}%(asctime)s{self.RESET}")
            # Wrap [%(fileinfo)s:%(lineno)d] with grey
            fmt = fmt.replace(
                "[%(fileinfo)s:%(lineno)d]",
                f"{self.GREY}[%(fileinfo)s:%(lineno)d]{self.RESET}",
            )

        # Call parent __init__ with potentially modified format string
        assert style in ("%", "{", "$"), "Unsupported style for logging.Formatter"
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        # Store original levelname to restore later (in case record is reused)
        orig_levelname = record.levelname

        # Only modify levelname - it needs dynamic color based on severity
        if (color_code := self.COLORS.get(record.levelname)) is not None:
            record.levelname = f"{color_code}{record.levelname}{self.RESET}"

        # Call parent format which will handle everything else
        msg = super().format(record)

        # Restore original levelname
        record.levelname = orig_levelname

        return msg


@lru_cache
def debug_once(logger: logging.Logger, msg: str, *args) -> None:
    from .distributed import get_world_group

    if not get_world_group().is_first_rank:
        return
    # Set the stacklevel to 3 to print the original caller's line info
    logger.debug(msg, *args, stacklevel=3)


@lru_cache
def info_once(logger: logging.Logger, msg: str, *args) -> None:
    from .distributed import get_world_group

    if not get_world_group().is_first_rank:
        return
    # Set the stacklevel to 3 to print the original caller's line info
    logger.info(msg, *args, stacklevel=3)


@lru_cache
def warning_once(logger: logging.Logger, msg: str, *args) -> None:
    from .distributed import get_world_group

    if not get_world_group().is_first_rank:
        return
    # Set the stacklevel to 3 to print the original caller's line info
    logger.warning(msg, *args, stacklevel=3)


def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter(
            "%(levelname)s %(asctime)s %(name)s:%(processName)s(%(process)s):%(threadName)s %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False

    if typing.TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            def info_once(self, msg, *args, **kwargs): ...
            def debug_once(self, msg, *args, **kwargs): ...
            def warning_once(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        logger.info_once = partial(info_once, logger)
        logger.debug_once = partial(debug_once, logger)
        logger.warning_once = partial(warning_once, logger)
        return logger
