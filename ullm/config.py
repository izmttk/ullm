import argparse
from dataclasses import dataclass


@dataclass
class EngineConfig:
    model: str
    gpu_memory_utilization: float
    max_bs: int
    tp_size: int
    pp_size: int
    nccl_port: int = 29500
    device_ids: list[int] | None = None
    enforce_eager: bool = False
    context_len: int = 2048

    log_level: str = "INFO"

    @staticmethod
    def from_args(args: argparse.Namespace) -> "EngineConfig":
        device_ids = (
            [int(i) for i in args.device_ids.split(",")] if args.device_ids else None
        )
        return EngineConfig(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_bs=args.max_bs,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            nccl_port=args.nccl_port,
            device_ids=device_ids,
            enforce_eager=args.enforce_eager,
            context_len=args.context_len,
        )


TIMEOUT_KEEP_ALIVE = 5  # seconds


@dataclass
class ServerConfig:
    host: str = ""
    port: int = 8000
    timeout_keep_alive: int = 5  # seconds

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ServerConfig":
        return ServerConfig(
            host=args.host,
            port=args.port,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        )
