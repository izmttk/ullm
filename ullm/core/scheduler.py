from collections import deque
from typing import Deque

from ..config import EngineConfig
from .common import (
    ScheduledCachedSequence,
    ScheduledNewSequence,
    SchedulerOutput,
    Sequence,
    SequenceStatus,
)
from .kv_cache import KVCacheManager


class Scheduler:
    """
    A lightweight scheduler.

    - Maintains two queues: waiting (prefill) and running (decode)
    - Integrates a KVCacheManager for prefix caching and eviction

    Notes
    -----
    - Policy: prefill-first. If there are waiting sequences, schedule a prefill batch.
      Otherwise schedule a decode batch from running sequences.
    - For KV cache:
        * On adding a new sequence, KV slots are reserved for the prompt length.
        * When appending new tokens (decode outputs), reserve 1 slot per token.
        * cache_sequence() is called to update the radix tree and deduplicate.
    - Updating cached_kv_len and appending new tokens should be done by the caller
      after model execution.
    """

    def __init__(
        self,
        config: EngineConfig,
        kv_cache_size: int,
    ):
        self.kv_cache_size = kv_cache_size
        self.max_bs = config.max_bs
        self.enable_async_scheduling = not config.disable_async_scheduling

        self.sequences: dict[str, Sequence] = {}  # All sequences by seq_id

        # Queues and registries
        self.waiting: Deque[Sequence] = deque()
        self.running: Deque[Sequence] = deque()

        # record preempted and finished seq_ids between scheduler steps
        self.preempted: set[str] = set()
        self.finished: set[str] = set()

        self.kv_manager = KVCacheManager(size=self.kv_cache_size)

    def add_sequence(self, seq: Sequence):
        """
        Add a new sequence to the scheduler.
        """
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)
        self.sequences[seq.seq_id] = seq

    def schedule(self) -> SchedulerOutput | None:
        """
        Pick a batch of sequences to run next, returning a list of sequences.
        """
        # batch: list[Sequence] = []

        scheduled_new_seqs: list[ScheduledNewSequence] = []

        # Prefill-first: schedule waiting sequences if any
        while self.waiting and len(scheduled_new_seqs) < self.max_bs:
            seq = self.waiting.popleft()
            # match prefix from KV cache on scheduling time
            self.kv_manager.match_prefix(seq)
            # Mark as running when scheduled for its first prefill
            seq.status = SequenceStatus.RUNNING
            self.alloc_kv_slots(seq)
            if self.enable_async_scheduling:
                seq.append_placeholder_tokens(1)
            scheduled_new_seqs.append(ScheduledNewSequence.from_sequence(seq))
            self.running.append(seq)
        if scheduled_new_seqs:
            finished_seq_ids = list(self.finished)
            self.finished.clear()
            preempted_seq_ids = list(self.preempted)
            self.preempted.clear()
            return SchedulerOutput(
                scheduled_new_seqs=scheduled_new_seqs,
                finished_seq_ids=finished_seq_ids,
                preempted_seq_ids=preempted_seq_ids,
            )

        scheduled_cached_seqs: list[ScheduledCachedSequence] = []
        popped_seqs: list[Sequence] = []
        # Otherwise, schedule decode from running queue
        while self.running and len(scheduled_cached_seqs) < self.max_bs:
            seq = self.running.popleft()
            popped_seqs.append(seq)
            if seq.num_tokens - seq.num_kv_indices == 0:
                # We now using num of input tokens to determine whether a sequence can be scheduled.
                # num_input_tokens = num_tokens - num_kv_indices
                # if num_input_tokens == 0, then this sequence no need to be scheduled.
                # if num_input_tokens > 0, we can schedule it.

                # This enable us to handle both cases of async scheduling and pipeline parallelism.
                # For async scheduling, we may have placeholder tokens in the sequence, meaning that
                # num_input_tokens > 0. In this case, we can schedule a sequence again even if
                # it has been scheduled just now.
                # For pipeline parallelism, a scheduled but not yet executed sequence must have
                # num_input_tokens == 0. In this case, we can avoid scheduling it again until
                # it has been executed and new tokens are appended.
                continue
            is_allocated = False
            while not is_allocated:
                try:
                    self.alloc_kv_slots(seq)
                    is_allocated = True
                except RuntimeError:
                    # Failed to allocate slots, preempt
                    if self.running:
                        # first preempt another running seq
                        seq_preempted = self.running.pop()
                        self.preempt(seq_preempted)
                    else:
                        # no other running seq, keep itself in running queue and break
                        self.running.appendleft(seq)
                        break
            if is_allocated:
                if self.enable_async_scheduling:
                    seq.append_placeholder_tokens(1)
                scheduled_cached_seqs.append(ScheduledCachedSequence.from_sequence(seq))

        # running queue may exists one seq (not enough slots for it)
        self.running.extendleft(reversed(popped_seqs))
        if scheduled_cached_seqs:
            finished_seq_ids = list(self.finished)
            self.finished.clear()
            preempted_seq_ids = list(self.preempted)
            self.preempted.clear()
            return SchedulerOutput(
                scheduled_cached_seqs=scheduled_cached_seqs,
                finished_seq_ids=finished_seq_ids,
                preempted_seq_ids=preempted_seq_ids,
            )

        return None

    def alloc_kv_slots(self, seq: Sequence):
        """
        Allocate KV slots for all uncached tokens in the sequence.
        """
        num_needed = seq.num_tokens - seq.num_kv_indices
        # if kv slots are already allocated for all tokens, skip allocation
        if num_needed <= 0:
            return
        new_slots = self.kv_manager.alloc_slots(num_needed)
        seq.append_new_kv_indices(new_slots)

    def free_kv_slots(self, seq: Sequence):
        """
        Free all KV slots associated with a sequence.
        """
        if seq.num_kv_indices == 0:
            return
        self.kv_manager.free_slots(seq.kv_indices.tolist())
        seq.clear_kv_indices()

    def preempt(self, seq: Sequence):
        """
        Preempt a running sequence back to waiting queue for prefill.
        """
        if seq.status == SequenceStatus.FINISHED:
            return
        if seq not in self.running:
            return
        self.running.remove(seq)
        self.preempted.add(seq.seq_id)
        self.free_kv_slots(seq)
        seq.status = SequenceStatus.WAITING
        if self.enable_async_scheduling:
            seq.clear_placeholder_tokens()
        self.waiting.appendleft(seq)  # Preempt to the front of waiting queue

    def get_sequence(self, sequence_id: str) -> Sequence | None:
        """
        Get a sequence by its ID.
        """
        return self.sequences.get(sequence_id, None)

    def update_sequence(self, seq: Sequence, new_token_id: int):
        """
        Update a sequence after model execution.
        """
        if seq.status != SequenceStatus.RUNNING:
            return

        if seq.num_placeholder_tokens > 0:
            seq.replace_placeholder_tokens([new_token_id])
        else:
            seq.append_new_tokens([new_token_id])
        seq.cache_new_kv_indices()

    def finish_sequence(self, seq: Sequence):
        """
        Mark a sequence as finished and cache its prefix in KV cache.
        """
        if seq.status == SequenceStatus.FINISHED:
            return

        if seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)

        if seq.status == SequenceStatus.RUNNING:
            self.running.remove(seq)
            self.finished.add(seq.seq_id)

        seq.status = SequenceStatus.FINISHED
        if self.enable_async_scheduling:
            seq.clear_placeholder_tokens()
        self.kv_manager.cache_sequence(seq)
        self.sequences.pop(seq.seq_id, None)

    def has_unfinished_sequences(self) -> bool:
        return bool(self.waiting or self.running)
