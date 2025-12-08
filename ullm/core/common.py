from dataclasses import dataclass, field
import enum

@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0

    max_tokens: int | None = None
    max_new_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    ignore_eos: bool = False
    eos_token_id: int = -1

class FinishReason(enum.Enum):
    STOP = enum.auto()
    LENGTH = enum.auto()

@dataclass
class GenerateOutput:
    token_str: str
    is_finished: bool
    finish_reason: FinishReason | None
    num_prompt_tokens: int
    num_generated_tokens: int

@dataclass
class EngineOutput:
    seq_id: str
    new_token_id: int
    is_finished: bool
    finish_reason: FinishReason | None
    num_prompt_tokens: int
    num_generated_tokens: int

class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()

@dataclass
class Sequence:
    seq_id: str
    status: SequenceStatus = SequenceStatus.WAITING
    num_tokens: int = 0
    prompt_len: int = 0
    # tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    kv_indices: list[int] = field(default_factory=list)
    cached_kv_len: int = 0
    # stream:  bool = False

class ForwardMode(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()

@dataclass
class ForwardBatch:
    forward_mode: ForwardMode
    num_seqs: int
    seqs: list[Sequence]
