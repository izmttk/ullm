
import numpy as np
import torch


class RequestState:
    def __init__(
        self,
        max_bs: int,
        context_len: int,
        device: torch.device,
    ):
        self.device = device
        self.seq_id_to_index: dict[str, int] = {}
        self.index_to_seq_id: dict[int, str] = {}
        self.free_indices = list(range(max_bs))
        
        self.new_token_ids = torch.zeros((max_bs, context_len),dtype=torch.long,device="cpu")
        self.num_new_tokens = torch.zeros((max_bs,), dtype=torch.long, device="cpu")
        self.kv_indices = torch.zeros((max_bs, context_len), dtype=torch.long, device="cpu")
        self.num_cached_kv_indices = torch.zeros((max_bs,), dtype=torch.long, device="cpu")
        self.num_new_kv_indices = torch.zeros((max_bs,), dtype=torch.long, device="cpu")
        self.num_kv_indices = torch.zeros((max_bs,), dtype=torch.long, device="cpu")
        
        # last_sampled_token 位于 GPU
        self.last_sampled_token = torch.zeros((max_bs,), dtype=torch.long, device=self.device)
        
    def add_sequence(
        self,
        seq_id: str,
        token_ids: np.ndarray,
        cached_kv_indices: np.ndarray,
        new_kv_indices: np.ndarray,
    ):
        req_idx = self.free_indices.pop()
        self.seq_id_to_index[seq_id] = req_idx
        self.index_to_seq_id[req_idx] = seq_id
        
        self.new_token_ids[req_idx, :len(token_ids)] = torch.tensor(token_ids, device="cpu")
        self.num_new_tokens[req_idx] = len(token_ids)
        kv_indices = np.concatenate((cached_kv_indices, new_kv_indices))
        self.kv_indices[req_idx, :len(kv_indices)] = torch.tensor(kv_indices, device="cpu")
        self.num_cached_kv_indices[req_idx] = len(cached_kv_indices)
        self.num_new_kv_indices[req_idx] = len(new_kv_indices)
        self.num_kv_indices[req_idx] = len(kv_indices)
        
    def remove_sequence(self, seq_id: str):
        if seq_id in self.seq_id_to_index:
            req_idx = self.seq_id_to_index.pop(seq_id)
            self.index_to_seq_id.pop(req_idx)
            self.free_indices.append(req_idx)
            # Optionally clear the input_ids and kv_indices for this index
            self.new_token_ids[req_idx].zero_()
            self.kv_indices[req_idx].zero_()
            self.num_new_tokens[req_idx].zero_()
            self.num_cached_kv_indices[req_idx].zero_()
            self.num_new_kv_indices[req_idx].zero_()
            self.num_kv_indices[req_idx].zero_()
    
    def update_sequence(
        self,
        seq_id: str,
        new_token_id: np.ndarray,
        new_kv_indices: np.ndarray,
    ):
        if seq_id in self.seq_id_to_index:
            req_idx = self.seq_id_to_index[seq_id]
            prev_num_kv = int(self.num_kv_indices[req_idx].item())
            
            self.new_token_ids[req_idx, :len(new_token_id)] = torch.tensor(new_token_id, device="cpu")
            self.num_new_tokens[req_idx] = len(new_token_id)
            self.kv_indices[req_idx, prev_num_kv:prev_num_kv+len(new_kv_indices)] = torch.tensor(new_kv_indices, device="cpu")
            self.num_cached_kv_indices[req_idx] = prev_num_kv
            self.num_new_kv_indices[req_idx] = len(new_kv_indices)
            self.num_kv_indices[req_idx] = prev_num_kv + len(new_kv_indices)
