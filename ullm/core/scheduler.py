from collections import deque
from typing import Deque

from .common import (
    Sequence,
    SequenceStatus,
    ForwardMode,
    ForwardBatch,
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
        kv_cache_size: int,
        max_bs: int = 32,
    ):
        self.kv_cache_size = kv_cache_size
        self.max_bs = max_bs
        
        self.sequences: dict[str, Sequence] = {}  # All sequences by seq_id

        # Queues and registries
        self.waiting: Deque[Sequence] = deque()
        self.running: Deque[Sequence] = deque()
        
        # To avoid scheduling the same sequence multiple times in one step
        # it's a subset of running queue
        self.scheduled: set[str] = set()

        self.kv_manager = KVCacheManager(size=self.kv_cache_size)

    def add_sequence(self, seq: Sequence):
        """
        Add a new sequence to the scheduler.
        """
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)
        self.sequences[seq.seq_id] = seq

    def schedule(self) -> ForwardBatch | None:
        """
        Pick a batch of sequences to run next, returning a list of sequences.
        """
        batch: list[Sequence] = []

        # Prefill-first: schedule waiting sequences if any
        while self.waiting and len(batch) < self.max_bs:
            seq = self.waiting.popleft()
            # Mark as running when scheduled for its first prefill
            seq.status = SequenceStatus.RUNNING
            self.alloc_kv_slots(seq)
            batch.append(seq)
            self.running.append(seq)
            self.scheduled.add(seq.seq_id)
        if batch:
            return ForwardBatch(
                forward_mode=ForwardMode.PREFILL,
                num_seqs=len(batch),
                seqs=batch,
            )

        popped_seqs: list[Sequence] = []
        # Otherwise, schedule decode from running queue
        while self.running and len(batch) < self.max_bs:
            seq = self.running.popleft()
            popped_seqs.append(seq)
            if seq.seq_id in self.scheduled:
                # Already scheduled in this step, skip
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
                batch.append(seq)
                self.scheduled.add(seq.seq_id)
        # running queue may exists one seq (not enough slots for it)
        self.running.extendleft(reversed(popped_seqs))
        if batch:
            return ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                num_seqs=len(batch),
                seqs=batch,
            )
        
        return None

    def alloc_kv_slots(self, seq: Sequence):
        """
        Allocate KV slots for all uncached tokens in the sequence.
        """
        num_needed = len(seq.token_ids) - len(seq.kv_indices)
        # if kv slots are already allocated for all tokens, skip allocation
        if num_needed <= 0:
            return
        new_slots = self.kv_manager.alloc_slots(num_needed)
        seq.kv_indices.extend(new_slots)

    def free_kv_slots(self, seq: Sequence):
        """
        Free all KV slots associated with a sequence.
        """
        if not seq.kv_indices:
            return
        self.kv_manager.free_slots(seq.kv_indices)
        seq.kv_indices.clear()
        seq.cached_kv_len = 0

    def preempt(self, seq: Sequence):
        """
        Preempt a running sequence back to waiting queue for prefill.
        """
        if seq.status == SequenceStatus.FINISHED:
            return
        if not seq in self.running:
            return
        self.running.remove(seq)
        if seq.seq_id in self.scheduled:
            self.scheduled.remove(seq.seq_id)
        self.free_kv_slots(seq)
        seq.status = SequenceStatus.WAITING
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
        
        self.scheduled.remove(seq.seq_id)
        seq.cached_kv_len = len(seq.kv_indices)
        seq.token_ids.append(new_token_id)
        seq.num_tokens = len(seq.token_ids)

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
            if seq.seq_id in self.scheduled:
                self.scheduled.remove(seq.seq_id)
        
        seq.status = SequenceStatus.FINISHED
        self.kv_manager.cache_sequence(seq)
        self.sequences.pop(seq.seq_id, None)

    def has_unfinished_sequences(self) -> bool:
        return bool(self.waiting or self.running)