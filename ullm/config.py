import argparse
from dataclasses import asdict, dataclass, field

from transformers import AutoConfig, GenerationConfig, PretrainedConfig

from .core.common import SamplingParams


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

    log_level: str = "info"
    profile: bool = False
    profile_dir: str = "./profiles"

    hf_config: PretrainedConfig = field(init=False)
    hf_generation_config: GenerationConfig | None = field(init=False)

    def try_get_generation_config(self) -> GenerationConfig | None:
        try:
            return GenerationConfig.from_pretrained(self.model)
        except OSError:  # Not found
            try:
                return GenerationConfig.from_model_config(self.hf_config)
            except OSError:  # Not found
                return None

    def __post_init__(self):
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.hf_generation_config = self.try_get_generation_config()

    def get_default_sampling_params(self) -> SamplingParams:
        if self.hf_generation_config is None:
            return SamplingParams()

        config = self.hf_generation_config.to_dict()
        default_params = asdict(SamplingParams())
        available_params = [
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
        ]

        default_sampling_params = {}
        for param, value in default_params.items():
            if param in available_params:
                default_sampling_params[param] = config.get(param, value)
            else:
                default_sampling_params[param] = value

        return SamplingParams(**default_sampling_params)

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
            log_level=args.log_level,
            profile=args.profile,
            profile_dir=args.profile_dir,
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
