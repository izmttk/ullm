# Graceful Shutdown and Windows Support

本文档说明了 ullm 项目中新增的优雅关闭机制和 Windows 平台支持。

## 功能概述

### 1. 优雅关闭 (Graceful Shutdown)

项目现在支持优雅关闭，可以正确处理以下信号：

- **SIGTERM**: 终止信号，常用于容器和系统服务
- **SIGINT**: 中断信号，通常由 Ctrl+C 触发
- **SIGBREAK** (仅 Windows): Windows 特有的中断信号

当收到这些信号时，系统会：

1. 停止接收新请求
2. 等待当前正在处理的请求完成
3. 清理所有工作进程/线程
4. 释放 GPU 资源和其他系统资源
5. 优雅退出

### 2. Windows 平台支持

项目现在完全支持 Windows 平台，通过以下改进：

- **线程模式**: 使用 Python 线程替代多进程，避免 Windows 下 CUDA 多进程的限制
- **平台兼容性**: 自动检测操作系统并应用适当的行为
- **信号处理**: 处理 Windows 特有的信号和限制

## 使用方法

### Linux/Unix 系统

在 Linux 或 Unix 系统上，可以使用默认的多进程模式：

```bash
python -m ullm.entrypoints.openai.api_server \
    --model Qwen3-0.6B \
    --gpu-memory-utilization 0.9 \
    --tp-size 2 \
    --pp-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

也可以选择使用线程模式：

```bash
python -m ullm.entrypoints.openai.api_server \
    --model Qwen3-0.6B \
    --use-threading \
    --tp-size 1 \
    --pp-size 1 \
    --host 0.0.0.0 \
    --port 8000
```

### Windows 系统

在 Windows 上，**强烈建议**使用线程模式：

```bash
python -m ullm.entrypoints.openai.api_server \
    --model Qwen3-0.6B \
    --use-threading \
    --tp-size 1 \
    --pp-size 1 \
    --host 0.0.0.0 \
    --port 8000
```

> **注意**: Windows 不支持多进程模式下的 CUDA，因此在 Windows 上必须使用 `--use-threading` 选项。

### 在代码中使用

```python
from ullm.llm import LLM, SamplingParams
import asyncio

async def main():
    # 在 Windows 上设置 use_threading=True
    llm = LLM(
        model="Qwen3-0.6B",
        gpu_memory_utilization=0.9,
        max_bs=50,
        tp_size=1,
        pp_size=1,
        use_threading=True,  # 在 Windows 上必需
    )
    
    await llm.ready()
    
    async for token in llm.generate(
        "Hello, world!",
        SamplingParams(max_new_tokens=50)
    ):
        print(token.token_str, end='', flush=True)
    
    # 优雅关闭
    llm.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## 技术细节

### 信号处理

在 `api_server.py` 中注册了信号处理器：

```python
def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _engine
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    if _engine is not None:
        _engine.shutdown()
    sys.exit(0)

# 在 lifespan 中注册
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
```

### 多进程 vs 线程模式

#### 多进程模式 (默认)

- 使用 `torch.multiprocessing`
- 每个 GPU rank 运行在独立进程中
- 更好的隔离性和稳定性
- **不支持 Windows**

#### 线程模式 (`--use-threading`)

- 使用 Python 标准库 `threading`
- 所有工作在同一进程的不同线程中运行
- 兼容 Windows
- 适合单 GPU 或小规模部署

### 平台特定代码

在 `utils.py` 中处理平台差异：

```python
def kill_itself_when_parent_died():
    """Set up process to be killed when parent dies (Linux only)."""
    if sys.platform == "linux":
        # Linux specific code
        PR_SET_PDEATHSIG = 1
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        except Exception as e:
            print(f"Warning: Failed to set PR_SET_PDEATHSIG: {e}")
    # On Windows, this is not available
```

在 `engine_client.py` 中：

```python
def _engine_loop(self):
    # Make child process session leader (Linux only)
    if not self.use_threading and sys.platform != "win32" and hasattr(os, 'setsid'):
        os.setsid()
    # ... rest of the code
```

### 关闭流程

1. **信号接收**: 主进程接收 SIGTERM/SIGINT 信号
2. **引擎关闭**: 调用 `engine.shutdown()`
3. **停止接收**: 通过队列发送 "shutdown" 消息
4. **等待完成**: 等待所有工作进程/线程完成当前任务
5. **资源清理**: 清理 CUDA 资源和分布式环境
6. **进程退出**: 所有进程/线程退出

## 限制和注意事项

1. **Windows + 多 GPU**: Windows 不支持多进程模式下的 CUDA，因此在 Windows 上只能使用单 GPU 或线程模式
2. **张量并行**: 线程模式下的张量并行支持可能受限
3. **性能**: 线程模式可能比多进程模式性能稍低，因为 GIL 的限制

## 故障排查

### 问题: 在 Windows 上无法启动

**解决方案**: 确保使用 `--use-threading` 选项

### 问题: Ctrl+C 无法停止服务器

**解决方案**: 
- 确保没有禁用信号处理
- 在 Windows 上，尝试使用 `Ctrl+Break` 或任务管理器

### 问题: 关闭时出现资源泄漏

**解决方案**: 
- 确保所有请求都已完成
- 检查日志中的错误信息
- 使用 `--enforce-eager` 禁用 CUDA graph 以简化清理

## 测试

运行测试脚本验证功能：

```bash
python test_graceful_shutdown.py
```

这将测试：
- 平台检测
- 信号处理
- 队列兼容性
- 线程模式功能

## 未来改进

- [ ] 更好的日志系统来追踪关闭过程
- [ ] 支持超时机制强制关闭
- [ ] 添加健康检查端点
- [ ] 改进 Windows 下的多 GPU 支持
