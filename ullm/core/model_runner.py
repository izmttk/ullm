import gc
import os
from dataclasses import dataclass

import torch
import tqdm
from transformers import PretrainedConfig

from ..config import EngineConfig
from ..core.input_batch import InputBatch
from ..distributed import all_gather, get_pp_group, get_pp_indices, get_tp_group
from ..layers.attention import AttentionBackend, attention_context
from ..layers.sampler import Sampler
from ..layers.utils import IntermediateTensors
from ..logger import init_logger
from ..model_loader import load_model
from ..models.registry import MODEL_REGISTRY
from .common import ForwardBatch, ForwardMode
from .cuda_graph import CUDAGraphManager
from .kv_cache import KVCachePool

logger = init_logger(__name__)


def set_cuda_arch():
    capability = torch.cuda.get_device_capability()
    arch = f"{capability[0]}.{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"


def get_model_config_per_gpu(
    hf_config: PretrainedConfig,
    tp_size: int,
    tp_rank: int,
    pp_size: int,
    pp_rank: int,
):
    hf_dtype = getattr(hf_config, "dtype", None) or getattr(
        hf_config, "torch_dtype", None
    )
    dtype = None
    if hf_dtype is None:
        hf_dtype = "float16"
    if isinstance(hf_dtype, torch.dtype):
        dtype = hf_dtype
    elif isinstance(hf_dtype, str):
        dtype = getattr(torch, hf_dtype, None)
        if not isinstance(dtype, torch.dtype):
            dtype = None
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {hf_dtype}")

    start_layer, end_layer = get_pp_indices(
        hf_config.num_hidden_layers, pp_rank, pp_size
    )
    num_layers = end_layer - start_layer
    num_heads = int(hf_config.num_attention_heads) // tp_size
    num_kv_heads = max(1, hf_config.num_key_value_heads // tp_size)

    if hasattr(hf_config, "head_dim"):
        head_dim = int(hf_config.head_dim)
    else:
        head_dim = int(hf_config.hidden_size // hf_config.num_attention_heads)

    return (
        dtype,
        start_layer,
        end_layer,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
    )


@dataclass
class ModelOutput:
    output: torch.Tensor
    copy_ready_event: torch.cuda.Event | None = None

    def get_output(self):
        if self.copy_ready_event is not None:
            self.copy_ready_event.synchronize()
        return self.output


class ModelRunner:
    def __init__(
        self,
        config: EngineConfig,
        rank: int,
        device: torch.device,
    ):
        self.config = config
        self.hf_config = self.config.hf_config

        self.model_path = config.model
        self.max_bs = config.max_bs
        self.enforce_eager = config.enforce_eager
        self.context_len = config.context_len

        self.rank = rank
        self.device = device

        set_cuda_arch()

    def initialize(self, gpu_memory_utilization: float = 0.9):
        # Initialize Model
        self.load_model()

        # Initialize KV Cache
        kv_cache_size = self.profile_kv_cache_size(gpu_memory_utilization)
        if self.rank == 0:
            logger.info_once(f"Max num tokens in kv cache: {kv_cache_size}")
        self.kv_cache_size = kv_cache_size
        self.kv_cache = KVCachePool(
            dtype=self.dtype,
            device=self.device,
            num_tokens=self.kv_cache_size,
            num_layers=self.num_layers,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )

        # Initialize Attention Backend
        self.attn_backend = AttentionBackend(
            kv_cache=self.kv_cache,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # Initialize input batch buffer
        self.input_batch = InputBatch(
            max_bs=self.max_bs,
            context_len=self.context_len,
            vocab_size=self.hf_config.vocab_size,
            device=self.device,
        )
        self.intermediate_tensors: IntermediateTensors | None = None
        if not get_pp_group().is_last_rank:
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_bs,
                dtype=self.dtype,
                device=self.device,
            )

        # Initialize CUDA Graph
        if not self.enforce_eager:
            self.cuda_graph = CUDAGraphManager()
            self.capture_graph()
        else:
            self.cuda_graph = None

    def load_model(self):
        architectures = getattr(self.hf_config, "architectures", [])
        ModelClass = None
        for arch in architectures:
            if arch in MODEL_REGISTRY:
                ModelClass = MODEL_REGISTRY[arch]
                break
        assert ModelClass is not None, (
            f"Model arch {self.hf_config.architectures} not supported."
        )

        logger.debug(
            f"Rank {self.rank} loading model {self.model_path} with type {ModelClass.__name__}."
        )

        torch_default_dtype = torch.get_default_dtype()

        (
            self.dtype,
            self.start_layer,
            self.end_layer,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        ) = get_model_config_per_gpu(
            self.hf_config,
            get_tp_group().size,
            get_tp_group().group_rank,
            get_pp_group().size,
            get_pp_group().group_rank,
        )
        logger.debug(
            f"Rank {self.rank} model config: {self.num_layers} layers, "
            f"{self.num_heads} heads, {self.num_kv_heads} kv heads, head dim {self.head_dim}, "
            f"dtype {self.dtype}, layers {self.start_layer}-{self.end_layer}."
        )

        torch.set_default_dtype(self.dtype)
        self.model = ModelClass(self.hf_config)  # type: ignore
        self.model.to(self.device)

        self.sampler = Sampler()

        load_model(self.model, self.model_path)

        torch.set_default_dtype(torch_default_dtype)

    def profile_kv_cache_size(self, gpu_memory_utilization: float = 0.9):
        cache_memsize_per_token = (
            self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * 2
            * self.dtype.itemsize
        )

        gc.collect()
        torch.cuda.empty_cache()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info(self.device)

        max_num_tokens = (
            int(free_gpu_memory * gpu_memory_utilization) // cache_memsize_per_token
        )
        max_num_tokens = torch.tensor([max_num_tokens], device=self.device)
        max_num_tokens = all_gather(max_num_tokens)
        max_num_tokens = int(max_num_tokens.min().item())

        gc.collect()
        torch.cuda.empty_cache()
        return max_num_tokens

    def prepare_input(
        self,
        batch: ForwardBatch,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        self.input_batch.apply_batch(batch)
        if self.intermediate_tensors is not None:
            assert intermediate_tensors is not None
            for name in self.intermediate_tensors.keys():
                tensor = intermediate_tensors[name]
                buffer = self.intermediate_tensors[name]
                buffer.zero_()
                buffer[: self.input_batch.num_new_tokens] = tensor

    @torch.inference_mode()
    def execute_model(
        self,
        batch: ForwardBatch,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelOutput | IntermediateTensors:
        assert hasattr(self, "model") and hasattr(self, "sampler"), (
            "Model and sampler must be loaded before execution."
        )
        assert hasattr(self, "kv_cache"), "KV Cache not initialized yet."
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None

        if batch.num_seqs == 0:
            if get_pp_group().is_last_rank:
                return ModelOutput(output=torch.empty((0,), device=self.device))
            else:
                return IntermediateTensors()
        self.prepare_input(batch, intermediate_tensors)

        # Forward pass
        if self.cuda_graph is not None and batch.forward_mode == ForwardMode.DECODE:
            hidden_states = self._execute_model_cuda_graph()
        else:
            hidden_states = self._execute_model_eager()

        if not get_pp_group().is_last_rank:
            assert isinstance(hidden_states, IntermediateTensors)
            # For mid-pipeline stages, return the hidden states.
            return hidden_states

        assert isinstance(hidden_states, torch.Tensor)
        # Compute logits
        logits_indices = self.input_batch.logits_indices[: self.input_batch.bs]
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states)

        # Sampling
        output_ids = self.sampler(
            logits,
            self.input_batch.temperatures[: self.input_batch.bs],
            self.input_batch.min_ps[: self.input_batch.bs],
            self.input_batch.top_ps[: self.input_batch.bs],
            self.input_batch.top_ks[: self.input_batch.bs],
        )

        return ModelOutput(output=output_ids)

    def _execute_model_cuda_graph(self) -> torch.Tensor | IntermediateTensors:
        assert self.cuda_graph is not None and hasattr(self, "intermediate_tensors"), (
            "CUDA Graph not initialized."
        )

        bs = self.input_batch.bs  # Only for decoding now
        padded_bs, graph_runner = self.cuda_graph.get_graph_runner(bs)

        assert bs == self.input_batch.num_new_tokens

        padded_input_ids = self.input_batch.input_ids[:bs]
        padded_positions = self.input_batch.positions[:bs]

        padded_intermediate_tensors = None
        if self.intermediate_tensors is not None:
            padded_intermediate_tensors = IntermediateTensors()
            for name in self.intermediate_tensors.keys():
                tensor = self.intermediate_tensors[name]
                padded_intermediate_tensors[name] = tensor[:padded_bs]

        attention_metadata = self.attn_backend.build_metadata_for_cuda_graph_replay(
            batch=self.input_batch,
            padded_bs=padded_bs,
        )

        padded_hidden_states = graph_runner.replay(
            padded_input_ids, padded_positions, padded_intermediate_tensors
        )

        if get_pp_group().is_last_rank:
            assert isinstance(padded_hidden_states, torch.Tensor)
            hidden_states = padded_hidden_states[:bs]
            return hidden_states
        else:
            assert isinstance(padded_hidden_states, IntermediateTensors)
            intermediate_tensors = IntermediateTensors()
            for name, tensor in padded_hidden_states.items():
                intermediate_tensors[name] = tensor[:bs]
            return intermediate_tensors

    def _execute_model_eager(self) -> torch.Tensor | IntermediateTensors:
        input_ids = self.input_batch.input_ids[: self.input_batch.num_new_tokens]
        positions = self.input_batch.positions[: self.input_batch.num_new_tokens]
        if self.intermediate_tensors is not None:
            intermediate_tensors = IntermediateTensors()
            for name in self.intermediate_tensors.keys():
                tensor = self.intermediate_tensors[name]
                intermediate_tensors[name] = tensor[: self.input_batch.num_new_tokens]
        else:
            intermediate_tensors = None

        attention_metadata = self.attn_backend.build_metadata(batch=self.input_batch)

        with attention_context(attention_metadata):
            hidden_states = self.model(input_ids, positions, intermediate_tensors)

        return hidden_states

    @torch.inference_mode()
    def capture_graph(self):
        assert self.cuda_graph is not None
        graph_bs = [1, 2, 4, 8] + list(range(16, self.max_bs, 16))
        if self.max_bs not in graph_bs:
            graph_bs.append(self.max_bs)
        logger.info_once(f"Capturing CUDA graphs for batch sizes: {graph_bs}")

        self.attn_backend.prepare_for_cuda_graph_io_buffers(
            max_bs=self.max_bs,
            context_len=self.context_len,
        )

        if get_pp_group().is_first_rank:
            self.intermediate_tensors = None
        else:
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_bs,
                dtype=self.dtype,
                device=self.device,
            )
        if get_pp_group().is_last_rank:
            self.hidden_states = torch.zeros(
                (self.max_bs, self.hf_config.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.hidden_states = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_bs,
                dtype=self.dtype,
                device=self.device,
            )

        compiled_model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")

        # Capture graphs for different batch sizes
        progress_bar = tqdm.tqdm(
            graph_bs, desc="Capturing CUDA Graphs", disable=(self.rank != 0)
        )
        for bs in progress_bar:
            attention_metadata = (
                self.attn_backend.build_metadata_for_cuda_graph_capture(bs=bs)
            )

            # prepare sliced inputs
            bs_input_ids = self.input_batch.input_ids[:bs]
            bs_positions = self.input_batch.positions[:bs]
            bs_intermediate_tensors = (
                IntermediateTensors(
                    {
                        name: tensor[:bs]
                        for name, tensor in self.intermediate_tensors.items()
                    }
                )
                if self.intermediate_tensors is not None
                else None
            )

            with attention_context(attention_metadata):
                graph_runner = self.cuda_graph.create_graph_runner(bs)
                outputs = graph_runner.capture(
                    compiled_model, bs_input_ids, bs_positions, bs_intermediate_tensors
                )
                if isinstance(self.hidden_states, torch.Tensor):
                    assert isinstance(outputs, torch.Tensor)
                    self.hidden_states[:bs] = outputs
                else:
                    assert isinstance(outputs, IntermediateTensors)
                    for name, buffer in self.hidden_states.items():
                        buffer[:bs] = outputs[name]

        torch.compiler.reset()
