# $\mathrm{\mu LLM}$ (micro-LLM)

A lightweight vLLM-like LLM inference engine with radix-tree based KV cache, and more.

该项目受 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/main) 启发，提供一个从零开始构建的 LLM 推理框架

程序的架构遵循了 [vLLM](https://github.com/vllm-project/vllm) v1 相似的组织安排，但是与 vLLM 不同的是，该项目的 KV 缓存系统使用的是 [SGLang](https://github.com/sgl-project/sglang) 的 Radix Cache 实现。

## Features

- 轻量但完整的代码实现
- FCFS 调度和持续批处理（Continuous Batching）
- [OpenAI 兼容 API](https://platform.openai.com/docs/api-reference/chat/create) 服务
- 基于 Radix Tree 的 Prefix Caching
- Flash Attention 算子支持（[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 实现）
- 张量并行（Tensor Parallelism）
- 流水线并行（Pipeline Parallelism）
- CUDA Graph 支持（仅 Decoding 阶段）
- 增量调度和有状态的 Worker
- Torch Profiler 支持

## Requirements

```plaintext
torch>=2.8.0
numpy
triton>=3.0.0
transformers>=4.51.0,<=4.57.3
fastapi>=0.95.0
uvicorn
flashinfer-python>=0.2.7
psutil
pyzmq>=25.0.0
msgspec
cloudpickle
tqdm
```

NOTE: 我发现当 nccl 版本 < 2.27.3 时，分布式通信和环境销毁可能会存在卡死的问题，建议升级 nvidia-nccl-cu12 至 2.27.3 及以上版本，torch 2.8.0 的依赖中已经包含该版本的 nccl。如果必须使用低版本 torch，请手动升级 nccl 版本 >= 2.27.3。

## Installation

```bash
git clone https://github.com/izmttk/ullm && cd ullm
pip install -e .
```

当然你也可以选择不安装，直接在项目根目录下运行服务。

## Quick Start

启动 API 服务

```plaintext
example:
python -m ullm.entrypoints.api_server --model Qwen3-0.6B --gpu-memory-utilization 0.9 --tp-size 2 --pp-size 2 --context-len 4096 --host 0.0.0.0 --port 8000

usage: api_server.py [-h] [--host HOST] [--port PORT] --model MODEL [--gpu-memory-utilization GPU_MEMORY_UTILIZATION] [--max-bs MAX_BS] [--tp-size TP_SIZE] [--pp-size PP_SIZE]
                     [--nccl-port NCCL_PORT] [--device-ids DEVICE_IDS] [--context-len CONTEXT_LEN] [--enforce-eager] [--log-level LOG_LEVEL] [--profile] [--profile-dir PROFILE_DIR]

LLM Distributed OpenAI-Compatible API Server

options:
  -h, --help            show this help message and exit
  --host HOST           Host name
  --port PORT           Port number
  --model MODEL         Model name
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        GPU memory utilization
  --max-bs MAX_BS       Maximum batch size
  --tp-size TP_SIZE     Tensor parallel size
  --pp-size PP_SIZE     Pipeline parallel size
  --nccl-port NCCL_PORT
                        NCCL port for distributed run
  --device-ids DEVICE_IDS
                        Comma-separated list of GPU device IDs to use
  --context-len CONTEXT_LEN
                        Max context length of the model
  --enforce-eager       Enforce eager execution, disable CUDA graph
  --log-level LOG_LEVEL
                        Log level for the engine
  --profile             Enable profiling support
  --profile-dir PROFILE_DIR
                        Directory to save profiling results
```

Offline Inference

```py
from ullm import LLM, SamplingParams, EngineConfig
import asyncio

async def main():
    config = EngineConfig(
        model="Qwen3-0.6B",
        gpu_memory_utilization=0.9,
        tp_size=2,
        pp_size=2,
        context_len=4096,
        enforce_eager=False,
    )
    llm = LLM(config)

    prompt = "Once upon a time"
    async for token in llm.generate(
        prompt,
        SamplingParams(
            max_new_tokens=50,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
    ):
        print(token, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

## Benchmarks

Experiment Environment:

- GPU: A100 40GB
- Model: Qwen3-0.6B
- Number of Requests: 256
- Prompt Length: random 100 ~ 1024
- Generation Length: random 100 ~ 1024
- Script: [bench.py](bench.py)

Results:

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------|----------|-----------------------|
| vLLM v0.11.0     | 133966        | 18.24    |  7343.96              |
| ours             | 133966        | 14.83    |  9032.37              |

## TODO

- [x] Graceful Shutdown
- [x] Better Logging System
- [x] Multi proc communication optimizations
- [x] More Configurable Options
- [x] Profiling
- [x] Increamental Batch Scheduling and Stateful Workers
- [ ] Further Improvements based on Profiling
- [ ] Overlap Scheduling
- [ ] Benchmark Metrics on API Server

Further development is still ongoing.
