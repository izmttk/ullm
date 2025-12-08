from contextlib import contextmanager
from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.kv_cache import KVCachePool
from ..core.common import ForwardMode, ForwardBatch
from ..core.cuda_graph import CUDAGraph

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

@dataclass
class AttentionMetadata:
    forward_mode: ForwardMode
    kv_cache: KVCachePool
    output_kv_indices: torch.Tensor
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper

class AttentionBackend:
    
    def __init__(
        self,
        kv_cache: KVCachePool,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device
    ):
        self.kv_cache = kv_cache
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # FlashInfer specific metadata 
        workspace_size = 512 * 1024 * 1024
        self.workspace_buffer = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device=self.device,
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="auto",
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
            backend="auto",
        )
        self.decode_wrappers_cuda_graph: dict[int, BatchDecodeWithPagedKVCacheWrapper] = {}

    def build_metadata(
        self,
        batch: ForwardBatch,
    ):
        page_size = 1
        seqlens_q = torch.tensor(
            [len(seq.token_ids) - seq.cached_kv_len for seq in batch.seqs],
            dtype=torch.int32,
            device=self.device
        )  # (max_bs,)
        seqlens_kv = torch.tensor(
            [len(seq.kv_indices) for seq in batch.seqs],
            dtype=torch.int32,
            device=self.device
        )  # (max_bs,)

        qo_indptr = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=self.device),
            torch.cumsum(seqlens_q, dim=0, dtype=torch.int32)
        ]) # (max_bs + 1,)
        
        paged_seqlens_kv = seqlens_kv // page_size + (seqlens_kv % page_size != 0).int() # (max_bs,)
        paged_kv_indptr = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=self.device),
            torch.cumsum(paged_seqlens_kv, dim=0, dtype=torch.int32)
        ])  # (max_bs + 1,)
        paged_kv_last_page_len = seqlens_kv % page_size
        paged_kv_last_page_len = torch.where(paged_kv_last_page_len == 0,
                                             page_size, paged_kv_last_page_len) # (max_bs,)
    
        paged_kv_indices = []
        for seq, seq_paged_len in zip(batch.seqs, paged_seqlens_kv):
            seq_paged_len = int(seq_paged_len.item())
            seq_paged_kv = torch.tensor(
                seq.kv_indices + [-1] * (seq_paged_len - len(seq.kv_indices)),
                dtype=torch.int32,
                device=self.device
            )
            paged_kv_indices.append(seq_paged_kv)
        paged_kv_indices = torch.cat(paged_kv_indices, dim=0)  # (num_total_pages,)

        wrapper = None
        if batch.forward_mode == ForwardMode.PREFILL:
            wrapper = self.prefill_wrapper
            self.prefill_wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                head_dim_vo=self.head_dim,
                page_size=page_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            wrapper = self.decode_wrapper
            self.decode_wrapper.plan(
                indptr=paged_kv_indptr,
                indices=paged_kv_indices,
                last_page_len=paged_kv_last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=page_size,
                q_data_type=self.dtype,
                kv_data_type=self.dtype
            )
        
        output_kv_indices = torch.cat(
            [torch.tensor(
                seq.kv_indices[seq.cached_kv_len:],
                dtype=torch.long,
                device=self.device
            ) for seq in batch.seqs],
            dim=0
        )

        assert wrapper is not None
        return AttentionMetadata(
            forward_mode=batch.forward_mode,
            kv_cache=self.kv_cache,
            output_kv_indices=output_kv_indices,
            wrapper=wrapper
        )
        
    def prepare_for_cuda_graph_capture(
        self,
        graph: CUDAGraph,
        max_bs: int,
        context_len: int,
    ):
        # capture mode, create max size buffers
        paged_kv_indptr_buffer = torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.device) * context_len
        graph.set_input_buffer("paged_kv_indptr_buffer", paged_kv_indptr_buffer)

        paged_kv_indices_buffer = torch.zeros((max_bs * context_len,), dtype=torch.int32, device=self.device)
        graph.set_input_buffer("paged_kv_indices_buffer", paged_kv_indices_buffer)
    
        paged_kv_last_page_len_buffer = torch.ones((max_bs,), dtype=torch.int32, device=self.device)
        graph.set_input_buffer("paged_kv_last_page_len_buffer", paged_kv_last_page_len_buffer)

        output_kv_indices_buffer = torch.zeros((max_bs,), dtype=torch.int64, device=self.device)
        graph.set_input_buffer("output_kv_indices_buffer", output_kv_indices_buffer)
            
    # build_metadata_for_cuda_graph_capture will create new decode_wrapper for each bs
    def build_metadata_for_cuda_graph_capture(
        self,
        graph: CUDAGraph,
        bs: int,
        context_len: int,
    ):
        paged_kv_indptr_buffer = graph.get_input_buffer("paged_kv_indptr_buffer")
        paged_kv_indices_buffer = graph.get_input_buffer("paged_kv_indices_buffer")
        paged_kv_last_page_len_buffer = graph.get_input_buffer("paged_kv_last_page_len_buffer")
        output_kv_indices_buffer = graph.get_input_buffer("output_kv_indices_buffer")
        
        paged_kv_indptr = paged_kv_indptr_buffer[:bs + 1]
        paged_kv_indices = paged_kv_indices_buffer[:bs * context_len]
        paged_kv_last_page_len = paged_kv_last_page_len_buffer[:bs]
        output_kv_indices = output_kv_indices_buffer[:bs]
        
        decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
            use_cuda_graph=True,
            backend="auto",
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len
        )
        decode_wrapper.plan(
            indptr=paged_kv_indptr,
            indices=paged_kv_indices,
            last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            q_data_type=self.dtype,
            kv_data_type=self.dtype
        )
        
        self.decode_wrappers_cuda_graph[bs]=decode_wrapper
        
        return AttentionMetadata(
            forward_mode=ForwardMode.DECODE,
            kv_cache=self.kv_cache,
            output_kv_indices=output_kv_indices,
            wrapper=decode_wrapper
        )
    
    # build_metadata_for_cuda_graph_replay will reuse decode_wrapper created during capture
    def build_metadata_for_cuda_graph_replay(
        self,
        graph: CUDAGraph,
        batch: ForwardBatch,
        padded_bs: int,
    ):
        # replay mode, fill in actual sizes
        bs = batch.num_seqs
        kv_len = sum(len(seq.kv_indices) for seq in batch.seqs)
        
        paged_kv_indptr_buffer = graph.get_input_buffer("paged_kv_indptr_buffer")
        paged_kv_indices_buffer = graph.get_input_buffer("paged_kv_indices_buffer")
        paged_kv_last_page_len_buffer = graph.get_input_buffer("paged_kv_last_page_len_buffer")
        output_kv_indices_buffer = graph.get_input_buffer("output_kv_indices_buffer")

        paged_kv_indptr = paged_kv_indptr_buffer[:padded_bs + 1]
        paged_kv_indices = paged_kv_indices_buffer[:kv_len]
        paged_kv_last_page_len = paged_kv_last_page_len_buffer[:padded_bs]
        output_kv_indices = output_kv_indices_buffer[:padded_bs]
        
        # padding kv len will be set to 1
        seqlens_kv = torch.tensor(
            [len(seq.kv_indices) for seq in batch.seqs] + [1] * (padded_bs - bs),
            dtype=torch.int32,
            device=self.device
        )  # (padded_bs,)
        paged_kv_indptr.copy_(
            torch.cat([
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(seqlens_kv, dim=0, dtype=torch.int32)
            ])  # (padded_bs + 1,)
        )
        
        kv_indices = [torch.tensor(
            seq.kv_indices,
            dtype=torch.int32,
            device=self.device
        ) for seq in batch.seqs]
        # fill kv indices buffer with 0 for padding sequences
        paged_kv_indices_buffer.fill_(0)
        paged_kv_indices.copy_(
            torch.cat(kv_indices, dim=0)  # (kv_len,)
        )
        
        # -1 means invalid, should be skipped for cache update.
        # this is important for case where num_seqs < graph_bs
        output_kv_indices_buffer.fill_(-1)
        output_kv_indices[:bs].copy_(
            torch.cat(
                [torch.tensor(
                    seq.kv_indices[seq.cached_kv_len:],
                    dtype=torch.long,
                    device=self.device
                ) for seq in batch.seqs],
                dim=0
            )
        )
        
        decode_wrapper = self.decode_wrappers_cuda_graph[padded_bs]
        decode_wrapper.plan(
            indptr=paged_kv_indptr,
            indices=paged_kv_indices,
            last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            q_data_type=self.dtype,
            kv_data_type=self.dtype
        )
        
        return AttentionMetadata(
            forward_mode=ForwardMode.DECODE,
            kv_cache=self.kv_cache,
            output_kv_indices=output_kv_indices,
            wrapper=decode_wrapper
        )

@contextmanager
def attention_kv_cache(model: nn.Module, metadata: AttentionMetadata):
    attn_modules: list[Attention] = []
    for module in model.modules():
        if isinstance(module, Attention):
            attn_modules.append(module)
            module.set_attention_metadata(metadata)
    yield
    for module in attn_modules:
        module.set_attention_metadata(None)

# flashinfer implemented attention
class Attention(nn.Module):
    def __init__(
        self,
        num_heads : int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.scaling = self.head_dim**-0.5 if scaling is None else scaling
        self.layer_id = layer_id

        self.attention_metadata: AttentionMetadata | None = None

    def set_attention_metadata(self, metadata: AttentionMetadata | None):
        self.attention_metadata = metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        save_kv_cache=True
    ):
        assert self.attention_metadata is not None

        cache_loc = self.attention_metadata.output_kv_indices
        q = q.contiguous()

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.attention_metadata.kv_cache.set_kv_cache(
                    self.layer_id,
                    cache_loc,
                    k.view(-1, self.num_kv_heads, self.head_dim),
                    v.view(-1, self.num_kv_heads, self.head_dim)
                )

        q = q.view(-1, self.num_heads, self.head_dim)
        k_cache, v_cache = self.attention_metadata.kv_cache.get_kv_cache(self.layer_id)
        k_cache = k_cache.view(-1, 1, self.num_kv_heads, self.head_dim)
        v_cache = v_cache.view(-1, 1, self.num_kv_heads, self.head_dim)

        # Call the wrapped function
        if self.attention_metadata.forward_mode == ForwardMode.PREFILL:
            self.attention_metadata.wrapper._sm_scale = self.scaling
            o = self.attention_metadata.wrapper.run(
                q, (k_cache, v_cache)
            )
        elif self.attention_metadata.forward_mode == ForwardMode.DECODE:
            self.attention_metadata.wrapper._sm_scale = self.scaling
            o = self.attention_metadata.wrapper.run(
                q, (k_cache, v_cache)
            )
        else:
            raise NotImplementedError

        return o.view(-1, self.num_heads * self.head_dim)
