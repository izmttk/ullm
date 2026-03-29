import torch

from ..logger import init_logger
from .common import ForwardMode, SchedulerOutput
from .req_state import RequestState

logger = init_logger(__name__)


class InputBatch:
    def __init__(
        self, max_bs: int, context_len: int, vocab_size: int, device: torch.device
    ):
        self.device = device
        # sequence id for each sequence in the batch
        # list of str with length bs
        self.seq_ids: list[str] = []
        # index in RequestState for each sequence in the batch
        self.req_idx_mapping = torch.zeros((max_bs,), dtype=torch.long, device=device)

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
        self.cu_seqlen_cpu = torch.zeros((max_bs + 1,), dtype=torch.long, device="cpu")

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
        self.cu_kv_seqlen_cpu = torch.zeros((max_bs + 1,), dtype=torch.long, device="cpu")

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
        self.num_cached_kv_indices = 0
        self.num_new_kv_indices = 0
        
        self._inflight_cpu_tensors = []  # keep references to cpu tensors to avoid premature deallocation

    def update_batch(
        self,
        sched_output: SchedulerOutput,
        request_state: RequestState,
    ):
        self.forward_mode = ForwardMode.PREFILL if sched_output.scheduled_new_seqs else ForwardMode.DECODE

        if self.forward_mode == ForwardMode.PREFILL:
            batch = sched_output.scheduled_new_seqs
        else:
            batch = sched_output.scheduled_cached_seqs

        self.bs = len(batch)
        
        self._inflight_cpu_tensors.clear()  # clear references to previous cpu tensors

        self.seq_ids = [seq.seq_id for seq in batch]
        req_idx_mapping_cpu = torch.tensor(
            [request_state.seq_id_to_index[seq.seq_id] for seq in batch],
            dtype=torch.long,
            device="cpu",
        ).pin_memory()
        self.req_idx_mapping[: self.bs].copy_(req_idx_mapping_cpu, non_blocking=True)

        # self.input_ids.zero_()
        # self.positions.zero_()
        # self.output_kv_indices.fill_(-1)
        # self.kv_indices.zero_()
        # self.cu_seqlen.zero_()
        # self.cu_kv_seqlen.zero_()
        # self.logits_indices.zero_()
        # self.temperatures.zero_()
        # self.min_ps.zero_()
        # self.top_ps.zero_()
        # self.top_ks.zero_()

        input_ids = torch.empty((0,), dtype=torch.long, device="cpu")
        positions = torch.empty((0,), dtype=torch.long, device="cpu")
        output_kv_indices = torch.empty((0,), dtype=torch.long, device="cpu")
        cu_seqlen = torch.zeros((self.bs + 1,), dtype=torch.long, device="cpu")
        kv_indices = torch.empty((0,), dtype=torch.long, device="cpu")
        cu_kv_seqlen = torch.zeros((self.bs + 1,), dtype=torch.long, device="cpu")
        logits_indices = torch.empty((self.bs,), dtype=torch.long, device="cpu")
        temperatures = torch.empty((self.bs,), dtype=torch.float, device="cpu")
        min_ps = torch.empty((self.bs,), dtype=torch.float, device="cpu")
        top_ps = torch.empty((self.bs,), dtype=torch.float, device="cpu")
        top_ks = torch.empty((self.bs,), dtype=torch.long, device="cpu")


        self.num_new_tokens = 0
        self.num_kv_indices = 0
        self.num_cached_kv_indices = 0
        self.num_new_kv_indices = 0
        cur_cu_seqlen = 0
        cur_cu_kv_seqlen = 0
        
        for i, seq in enumerate(batch):
            seq_id = seq.seq_id
            req_idx = request_state.seq_id_to_index[seq_id]
            
            num_new_tokens = int(request_state.num_new_tokens[req_idx].item())
            num_kv_indices = int(request_state.num_kv_indices[req_idx].item())
            num_cached_kv_indices = int(request_state.num_cached_kv_indices[req_idx].item())
            num_new_kv_indices = int(request_state.num_new_kv_indices[req_idx].item())
            
            seq_input_ids = request_state.new_token_ids[req_idx][:num_new_tokens]
            seq_positions = torch.arange(
                num_cached_kv_indices, num_cached_kv_indices + num_new_tokens,dtype=torch.long, device="cpu")
            input_ids = torch.cat((input_ids, seq_input_ids))
            positions = torch.cat((positions, seq_positions))
            
            seq_output_kv_indices = request_state.kv_indices[req_idx][num_cached_kv_indices:num_kv_indices]
            output_kv_indices = torch.cat((output_kv_indices, seq_output_kv_indices))

            cur_cu_seqlen += len(seq_input_ids)
            cu_seqlen[i + 1] = cur_cu_seqlen

            seq_kv_indices = request_state.kv_indices[req_idx][:num_kv_indices]
            kv_indices = torch.cat((kv_indices, seq_kv_indices))
            cur_cu_kv_seqlen += len(seq_kv_indices)
            cu_kv_seqlen[i + 1] = cur_cu_kv_seqlen

            logits_indices[i] = cur_cu_seqlen - 1  # typically compute logits for the last token
            temperatures[i] = seq.sampling_params.temperature
            min_ps[i] = seq.sampling_params.min_p
            top_ps[i] = seq.sampling_params.top_p
            
            if seq.sampling_params.top_k == -1:
                top_k = self.vocab_size
            else:
                top_k = min(seq.sampling_params.top_k, self.vocab_size)
            top_ks[i] = top_k
            
            self.num_new_tokens += num_new_tokens
            self.num_kv_indices += num_kv_indices
            self.num_cached_kv_indices += num_cached_kv_indices
            self.num_new_kv_indices += num_new_kv_indices
        
        
        # launch some necessary gpu kernels
        input_ids_cpu = input_ids.pin_memory()

        input_ids = input_ids_cpu.to(self.device, non_blocking=True)
        # Here we assume each request only has one new token, which is typically the case for decode steps.
        last_token = request_state.last_sampled_token[self.req_idx_mapping[: self.bs]]
        input_ids = input_ids.where(input_ids != -1, last_token)
        self.input_ids[: self.num_new_tokens] = input_ids
        
        positions_cpu = positions.pin_memory()
        output_kv_indices_cpu = output_kv_indices.pin_memory()
        kv_indices_cpu = kv_indices.pin_memory()
        logits_indices_cpu = logits_indices.pin_memory()
        temperatures_cpu = temperatures.pin_memory()
        min_ps_cpu = min_ps.pin_memory()
        top_ps_cpu = top_ps.pin_memory()
        top_ks_cpu = top_ks.pin_memory()

        self.cu_seqlen_cpu[: self.bs + 1] = cu_seqlen
        self.cu_kv_seqlen_cpu[: self.bs + 1] = cu_kv_seqlen

        self.positions[: self.num_new_tokens].copy_(positions_cpu, non_blocking=True)
        self.output_kv_indices[: self.num_new_kv_indices].copy_(output_kv_indices_cpu, non_blocking=True)
        self.kv_indices[: self.num_kv_indices].copy_(kv_indices_cpu, non_blocking=True)
        self.logits_indices[: self.bs].copy_(logits_indices_cpu, non_blocking=True)
        self.temperatures[: self.bs].copy_(temperatures_cpu, non_blocking=True)
        self.min_ps[: self.bs].copy_(min_ps_cpu, non_blocking=True)
        self.top_ps[: self.bs].copy_(top_ps_cpu, non_blocking=True)
        self.top_ks[: self.bs].copy_(top_ks_cpu, non_blocking=True)
        
        self._inflight_cpu_tensors.extend([
            req_idx_mapping_cpu,
            input_ids_cpu,
            positions_cpu,
            output_kv_indices_cpu,
            kv_indices_cpu,
            logits_indices_cpu,
            temperatures_cpu,
            min_ps_cpu,
            top_ps_cpu,
            top_ks_cpu,
        ])
