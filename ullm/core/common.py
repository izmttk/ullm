import enum
from dataclasses import dataclass, field

import numpy as np


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
class EngineStepResult:
    seq_id: str
    new_token_id: int
    is_finished: bool
    finish_reason: FinishReason | None
    num_prompt_tokens: int
    num_generated_tokens: int


class EngineDeadError(Exception):
    """
    This exception is used to inform fastapi handlers that engine is dead.
    """

    MESSAGE = "Engine process has exited unexpectedly."

    def __init__(self, message: str = MESSAGE, *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


@dataclass
class Sequence:
    seq_id: str
    status: SequenceStatus = SequenceStatus.WAITING
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    """
    token_ids could be separated into prompt and generated tokens,
    +---------------------------+--------------------------+
    | prompt_token_ids          | generated_token_ids      |
    +---------------------------+--------------------------+
    |<--- num_prompt_tokens --->|<- num_generated_tokens ->|
    |<------------------- num_tokens --------------------->|
    """
    token_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    num_prompt_tokens: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def prompt_token_ids(self) -> np.ndarray:
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def generated_token_ids(self) -> np.ndarray:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_generated_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    """
    kv_indices mentions the allocated KV cache slots for this sequence.
    Note that len(kv_indices) could be smaller than len(token_ids), because
    some tokens are newly added/generated and have not been allocated KV slots yet.
    +-------------------------------+------------------------+
    | cached_kv_indices             | new_kv_indices         |
    +-------------------------------+------------------------+
    |<--- num_cached_kv_indices --->|<- num_new_kv_indices ->|
    |<----------------- num_kv_indices --------------------->|
    """
    cached_kv_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    new_kv_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )

    @property
    def kv_indices(self) -> np.ndarray:
        return np.concatenate((self.cached_kv_indices, self.new_kv_indices))

    @property
    def num_kv_indices(self) -> int:
        return self.num_cached_kv_indices + self.num_new_kv_indices

    @property
    def num_cached_kv_indices(self) -> int:
        return len(self.cached_kv_indices)

    @property
    def num_new_kv_indices(self) -> int:
        return len(self.new_kv_indices)

    @staticmethod
    def create(
        seq_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> "Sequence":
        return Sequence(
            seq_id=seq_id,
            token_ids=np.array(prompt_token_ids, dtype=np.int32),
            num_prompt_tokens=len(prompt_token_ids),
            sampling_params=sampling_params,
        )

    def set_cached_kv_indices(self, cached_kv_indices: list[int]):
        assert len(cached_kv_indices) <= self.num_tokens
        self.cached_kv_indices = np.array(cached_kv_indices, dtype=np.int32)

    def set_new_kv_indices(self, new_kv_indices: list[int]):
        assert len(new_kv_indices) == self.num_tokens - self.num_cached_kv_indices
        self.new_kv_indices = np.array(new_kv_indices, dtype=np.int32)

    def cache_new_kv_indices(self):
        self.cached_kv_indices = np.concatenate(
            (self.cached_kv_indices, self.new_kv_indices)
        )
        self.new_kv_indices = np.array([], dtype=np.int32)

    def append_new_tokens(self, new_token_ids: list[int]):
        self.token_ids = np.concatenate(
            (self.token_ids, np.array(new_token_ids, dtype=np.int32))
        )

    def clear_kv_indices(self):
        self.cached_kv_indices = np.array([], dtype=np.int32)
        self.new_kv_indices = np.array([], dtype=np.int32)


class ForwardMode(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


@dataclass
class ForwardBatch:
    forward_mode: ForwardMode
    num_seqs: int
    seqs: list[Sequence]
