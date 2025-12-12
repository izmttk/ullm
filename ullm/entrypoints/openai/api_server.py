"""
This file is adapted from
https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import signal
import sys
import uvicorn
import fastapi
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from ullm.llm import LLM
from .serving_chat import OpenAIServingChat
from .serving_completion import OpenAIServingCompletion
from ..utils import with_cancellation

TIMEOUT_KEEP_ALIVE = 5  # seconds

# Global engine reference for signal handler
_engine = None


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _engine
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    if _engine is not None:
        _engine.shutdown()
    sys.exit(0)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    global _engine
    device_ids = [int(i) for i in args.device_ids.split(",")] if args.device_ids else None
    engine = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_bs=args.max_bs,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        nccl_port=args.nccl_port,
        device_ids=device_ids,
        enforce_eager=args.enforce_eager,
        context_len=args.context_len,
        use_threading=args.use_threading,
    )
    await engine.ready()
    _engine = engine

    # Register signal handlers for graceful shutdown
    # Note: Using signal.signal() here is simple but not fully thread-safe with asyncio.
    # For production use, consider using asyncio.add_signal_handler() on Unix systems.
    # This current approach works for most use cases and is cross-platform compatible.
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    # On Windows, also handle SIGBREAK if available
    if sys.platform == "win32":
        try:
            signal.signal(signal.SIGBREAK, _signal_handler)  # type: ignore
        except AttributeError:
            pass  # SIGBREAK not available in this Python build

    app.state.model_name = args.model
    app.state.serving_chat = OpenAIServingChat(engine, args.model)
    app.state.serving_completion = OpenAIServingCompletion(engine, args.model)
    
    yield
    
    print("Shutting down engine...")
    engine.shutdown()
    _engine = None

app = fastapi.FastAPI(lifespan=lifespan)

def create_error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code,
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: fastapi.Request, exc):  # pylint: disable=unused-argument
    return create_error_response(400, str(exc))

@app.get("/v1/models")
async def show_models(request: fastapi.Request):
    model_card = ModelCard(id=request.app.state.model_name)
    return ModelList(data=[model_card])

@app.post("/v1/completions")
@with_cancellation
async def create_completion(request: CompletionRequest, raw_request: fastapi.Request):
    serving_completion = raw_request.app.state.serving_completion
    return await serving_completion.create_completion(request)

@app.post("/v1/chat/completions")
@with_cancellation
async def create_chat_completion(request: ChatCompletionRequest, raw_request: fastapi.Request):
    serving_chat = raw_request.app.state.serving_chat
    return await serving_chat.create_chat_completion(request)

def run_server(args: argparse.Namespace):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Distributed OpenAI-Compatible API Server"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Host name")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument("--max-bs", type=int, default=50, help="Maximum batch size")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--nccl-port", type=int, default=29500, help="NCCL port for distributed run"
    )
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU device IDs to use",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=2048,
        help="Max context length of the model",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager execution, disable CUDA graph",
    )
    parser.add_argument(
        "--use-threading",
        action="store_true",
        help="Use threading instead of multiprocessing (recommended for Windows)",
    )
    args = parser.parse_args()
    
    run_server(args)

