import numpy as np
import torch

from ..logger import init_logger
from .common import ForwardBatch, ForwardMode

logger = init_logger(__name__)


class InputBatch:
    def __init__(
        self, max_bs: int, context_len: int, vocab_size: int, device: torch.device
    ):
        # sequence id for each sequence in the batch
        # list of str with length bs
        self.seq_ids: list[str] = []

        # Flattened input ids and positions for the current forward step, concatenated across all sequences in the batch.
        # size: (num_new_tokens,)
        self.input_ids = torch.zeros(
            (max_bs * context_len,), dtype=torch.long, device=device
        )
        # The position of each input token in the original sequence, used for positional embeddings.
        # size: (num_new_tokens,)
        self.positions = torch.zeros(
            (max_bs * context_len,),
            dtype=torch.long,
            device=device,
        )
        # size: (max_bs + 1,)
        self.cu_seqlen = torch.zeros((max_bs + 1,), dtype=torch.long, device=device)

        # Note: output_kv_indices concatenates new kv indices for all sequences in the batch, which
        # will be used to store new kv before attention computation.
        # kv_indices is used in attention computation to index into the kv cache, it includes both cached and new kv indices.
        # -1 means invalid index and will be ignored kv cache storage.
        # size: (num_new_kv_indices,)
        self.output_kv_indices = torch.full(
            (max_bs * context_len,), -1, dtype=torch.long, device=device
        )

        # size (num_kv_indices,)
        self.kv_indices = torch.zeros(
            (max_bs * context_len,), dtype=torch.long, device=device
        )
        # size: (max_bs + 1,)
        self.cu_kv_seqlen = torch.zeros((max_bs + 1,), dtype=torch.long, device=device)

        # For each sequence in the batch, the index of the token to compute logits for in the current forward step.
        # Typically indices of the last token for each sequence.
        # 提前计算每个序列的最后一个 token 在输入中的位置，可以避免隐式同步
        # size: (max_bs,)
        self.logits_indices = torch.zeros((max_bs,), dtype=torch.long, device=device)

        # sampling parameters for each sequence in the batch
        # size: (max_bs,)
        self.temperatures = torch.zeros((max_bs,), dtype=torch.float, device=device)
        # size: (max_bs,)
        self.min_ps = torch.zeros((max_bs,), dtype=torch.float, device=device)
        # size: (max_bs,)
        self.top_ps = torch.zeros((max_bs,), dtype=torch.float, device=device)
        # size: (max_bs,)
        self.top_ks = torch.zeros((max_bs,), dtype=torch.long, device=device)

        self.forward_mode: ForwardMode | None = None
        self.max_bs = max_bs
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.bs = 0
        self.num_new_tokens = 0
        self.num_kv_indices = 0
        self.num_new_kv_indices = 0

    def apply_batch(self, batch: ForwardBatch):
        self.forward_mode = batch.forward_mode
        self.bs = batch.num_seqs

        self.seq_ids = [seq.seq_id for seq in batch.seqs]
        self.num_new_tokens = sum(
            seq.num_tokens - seq.num_cached_kv_indices for seq in batch.seqs
        )
        self.num_kv_indices = sum(seq.num_kv_indices for seq in batch.seqs)
        self.num_new_kv_indices = sum(seq.num_new_kv_indices for seq in batch.seqs)

        self.input_ids.zero_()
        self.positions.zero_()
        self.output_kv_indices.fill_(-1)
        self.kv_indices.zero_()
        self.cu_seqlen.zero_()
        self.cu_kv_seqlen.zero_()
        self.logits_indices.zero_()
        self.temperatures.zero_()
        self.min_ps.zero_()
        self.top_ps.zero_()
        self.top_ks.zero_()

        input_ids: list[int] = []
        positions: list[int] = []
        output_kv_indices: list[int] = []
        cu_seqlen: list[int] = [0]
        kv_indices: list[int] = []
        cu_kv_seqlen: list[int] = [0]
        logits_indices: list[int] = []
        temperatures: list[float] = []
        min_ps: list[float] = []
        top_ps: list[float] = []
        top_ks: list[int] = []

        cur_cu_seqlen = 0
        cur_cu_kv_seqlen = 0
        for seq in batch.seqs:
            seq_input_ids = seq.token_ids[seq.num_cached_kv_indices :]
            seq_positions = np.arange(
                seq.num_cached_kv_indices, seq.num_tokens, dtype=np.int64
            )
            input_ids.extend(seq_input_ids.tolist())
            positions.extend(seq_positions.tolist())
            output_kv_indices.append(seq.new_kv_indices.tolist())

            cur_cu_seqlen += len(seq_input_ids)
            cu_seqlen.append(cur_cu_seqlen)

            seq_kv_indices = seq.kv_indices
            kv_indices.extend(seq_kv_indices.tolist())
            cur_cu_kv_seqlen += seq.num_kv_indices
            cu_kv_seqlen.append(cur_cu_kv_seqlen)

            logits_indices.append(cur_cu_seqlen - 1)

            temperatures.append(seq.sampling_params.temperature)
            min_ps.append(seq.sampling_params.min_p)
            top_ps.append(seq.sampling_params.top_p)
            if seq.sampling_params.top_k == -1:
                top_k = self.vocab_size
            else:
                top_k = min(seq.sampling_params.top_k, self.vocab_size)
            top_ks.append(top_k)

        self.input_ids[: self.num_new_tokens] = torch.tensor(
            input_ids, dtype=torch.long, device="cpu"
        )
        self.positions[: self.num_new_tokens] = torch.tensor(
            positions, dtype=torch.long, device="cpu"
        )
        self.output_kv_indices[: self.num_new_kv_indices] = torch.tensor(
            output_kv_indices, dtype=torch.long, device="cpu"
        )
        self.kv_indices[: self.num_kv_indices] = torch.tensor(
            kv_indices, dtype=torch.long, device="cpu"
        )
        self.cu_seqlen[: self.bs + 1] = torch.tensor(
            cu_seqlen, dtype=torch.long, device="cpu"
        )
        self.cu_kv_seqlen[: self.bs + 1] = torch.tensor(
            cu_kv_seqlen, dtype=torch.long, device="cpu"
        )
        self.logits_indices[: self.bs] = torch.tensor(
            logits_indices, dtype=torch.long, device="cpu"
        )
        self.temperatures[: self.bs] = torch.tensor(
            temperatures, dtype=torch.float, device="cpu"
        )
        self.min_ps[: self.bs] = torch.tensor(min_ps, dtype=torch.float, device="cpu")
        self.top_ps[: self.bs] = torch.tensor(top_ps, dtype=torch.float, device="cpu")
        self.top_ks[: self.bs] = torch.tensor(top_ks, dtype=torch.long, device="cpu")
