from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from ..core.common import ForwardMode
from ..core.input_batch import InputBatch
from ..core.kv_cache import KVCachePool


@dataclass
class AttentionMetadata:
    forward_mode: ForwardMode
    kv_cache: KVCachePool
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper

    output_kv_indices: torch.Tensor


class AttentionBackend:
    def __init__(
        self,
        kv_cache: KVCachePool,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
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
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
        )
        self.decode_wrappers_cuda_graph: dict[
            int, BatchDecodeWithPagedKVCacheWrapper
        ] = {}

    def build_metadata(
        self,
        batch: InputBatch,
    ):
        qo_indptr = batch.cu_seqlen[: batch.bs + 1].to(
            dtype=torch.int32, device="cpu"
        )  # (bs + 1,)
        paged_kv_indptr = batch.cu_kv_seqlen[: batch.bs + 1].to(
            dtype=torch.int32, device="cpu"
        )  # (bs + 1,)
        paged_kv_last_page_len = torch.ones(
            batch.bs, dtype=torch.int32, device="cpu"
        )  # (bs,)
        paged_kv_indices = batch.kv_indices[: batch.num_kv_indices].to(
            dtype=torch.int32, device=self.device
        )  # (num_kv_indices,)

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
                page_size=1,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
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
                page_size=1,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )

        output_kv_indices = batch.output_kv_indices[: batch.num_new_kv_indices]
        assert wrapper is not None and batch.forward_mode is not None
        return AttentionMetadata(
            forward_mode=batch.forward_mode,
            kv_cache=self.kv_cache,
            wrapper=wrapper,
            output_kv_indices=output_kv_indices,
        )

    def prepare_for_cuda_graph_io_buffers(
        self,
        max_bs: int,
        context_len: int,
    ):
        self.context_len = context_len
        # capture mode, create max size buffers
        self.paged_kv_indptr_buffer = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.paged_kv_indices_buffer = torch.zeros(
            max_bs * context_len, dtype=torch.int32, device=self.device
        )
        self.paged_kv_last_page_len_buffer = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.output_kv_indices_buffer = torch.zeros(
            max_bs, dtype=torch.int64, device=self.device
        )

    # build_metadata_for_cuda_graph_capture will create new decode_wrapper for each bs
    def build_metadata_for_cuda_graph_capture(self, bs: int):
        paged_kv_indptr_buffer = self.paged_kv_indptr_buffer[: bs + 1]
        paged_kv_indices_buffer = self.paged_kv_indices_buffer[: bs * self.context_len]
        paged_kv_last_page_len_buffer = self.paged_kv_last_page_len_buffer[:bs]

        decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=False,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=paged_kv_indptr_buffer,
            paged_kv_indices_buffer=paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
        )

        paged_kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device="cpu")
        # paged_kv_indices is placed in cuda decive for reducing a h2d copy
        paged_kv_indices = paged_kv_indices_buffer
        paged_kv_last_page_len = torch.ones(bs, dtype=torch.int32, device="cpu")
        output_kv_indices = self.output_kv_indices_buffer[:bs]

        decode_wrapper.plan(
            indptr=paged_kv_indptr,
            indices=paged_kv_indices,
            last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

        self.decode_wrappers_cuda_graph[bs] = decode_wrapper

        return AttentionMetadata(
            forward_mode=ForwardMode.DECODE,
            kv_cache=self.kv_cache,
            wrapper=decode_wrapper,
            output_kv_indices=output_kv_indices,
        )

    # build_metadata_for_cuda_graph_replay will reuse decode_wrapper created during capture
    def build_metadata_for_cuda_graph_replay(
        self,
        batch: InputBatch,
        padded_bs: int,
    ):
        # replay mode, fill in actual sizes
        bs = batch.bs

        # padding kv len will be set to 1
        paged_kv_indptr = torch.cat(
            [
                batch.cu_kv_seqlen[: bs + 1].to(dtype=torch.int32, device="cpu"),
                torch.arange(
                    batch.num_kv_indices + 1,
                    batch.num_kv_indices + 1 + padded_bs - bs,
                    dtype=torch.int32,
                    device="cpu",
                ),
            ]
        )  # (padded_bs + 1,)

        paged_kv_indices = torch.cat(
            [
                batch.kv_indices[: batch.num_kv_indices].to(
                    dtype=torch.int32, device=self.device
                ),
                torch.zeros((padded_bs - bs), dtype=torch.int32, device=self.device),
            ],
        )  # (kv_len + padded_bs - bs,)

        # Note: flashinfer requires last_page_len > 0:
        # "The last_page_len of each request must be greater than zero, and less than or equal to page_size."
        paged_kv_last_page_len = torch.ones(
            padded_bs, dtype=torch.int32, device="cpu"
        )  # (padded_bs,)

        output_kv_indices = self.output_kv_indices_buffer[:padded_bs]
        # -1 means invalid, should be skipped for cache update.
        # this is important for case where num_seqs < graph_bs
        output_kv_indices.fill_(-1)
        output_kv_indices[:bs] = batch.output_kv_indices[:bs]

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
            kv_data_type=self.dtype,
        )

        return AttentionMetadata(
            forward_mode=ForwardMode.DECODE,
            kv_cache=self.kv_cache,
            wrapper=decode_wrapper,
            output_kv_indices=output_kv_indices,
        )


@dataclass
class AttentionContext:
    attention_metadata: AttentionMetadata | None = None


_attention_context = AttentionContext()


@contextmanager
def attention_context(attention_metadata: AttentionMetadata):
    _attention_context.attention_metadata = attention_metadata
    yield
    _attention_context.attention_metadata = None


# flashinfer implemented attention
class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.scaling = self.head_dim**-0.5 if scaling is None else scaling
        self.layer_id = layer_id

        self.attention_context = _attention_context

    @torch.compiler.disable
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, save_kv_cache=True
    ):
        attention_metadata = self.attention_context.attention_metadata
        assert attention_metadata is not None

        cache_loc = attention_metadata.output_kv_indices
        q = q.contiguous()

        if k is not None:
            assert v is not None
            if save_kv_cache:
                attention_metadata.kv_cache.set_kv_cache(
                    self.layer_id,
                    cache_loc,
                    k.view(-1, self.num_kv_heads, self.head_dim),
                    v.view(-1, self.num_kv_heads, self.head_dim),
                )

        q = q.view(-1, self.num_heads, self.head_dim)
        k_cache, v_cache = attention_metadata.kv_cache.get_kv_cache(self.layer_id)
        k_cache = k_cache.view(-1, 1, self.num_kv_heads, self.head_dim)
        v_cache = v_cache.view(-1, 1, self.num_kv_heads, self.head_dim)

        # Call the wrapped function
        if attention_metadata.forward_mode == ForwardMode.PREFILL:
            attention_metadata.wrapper._sm_scale = self.scaling
            o = attention_metadata.wrapper.run(q, (k_cache, v_cache))
        elif attention_metadata.forward_mode == ForwardMode.DECODE:
            attention_metadata.wrapper._sm_scale = self.scaling
            o = attention_metadata.wrapper.run(q, (k_cache, v_cache))
        else:
            raise NotImplementedError

        return o.view(-1, self.num_heads * self.head_dim)
