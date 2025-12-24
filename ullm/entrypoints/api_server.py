"""
This file is adapted from
https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
"""

import argparse
import gc
import logging
from contextlib import asynccontextmanager
from http import HTTPStatus

import fastapi
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import EngineConfig, ServerConfig
from ..llm import LLM
from .protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from .serving_chat import OpenAIServingChat
from .serving_completion import OpenAIServingCompletion
from .utils import with_cancellation


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    engine = None
    try:
        config = EngineConfig.from_args(args)
        engine = LLM(config)

        app.state.model_name = args.model
        app.state.serving_chat = OpenAIServingChat(engine, args.model)
        app.state.serving_completion = OpenAIServingCompletion(engine, args.model)

        yield
    finally:
        if hasattr(app.state, "model_name"):
            del app.state.model_name
        if hasattr(app.state, "serving_chat"):
            del app.state.serving_chat
        if hasattr(app.state, "serving_completion"):
            del app.state.serving_completion
        if engine:
            await engine.shutdown()
            del engine
        gc.collect()


app = fastapi.FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: fastapi.Request, exc):  # pylint: disable=unused-argument
    return JSONResponse(
        ErrorResponse(
            message=str(exc),
            type="invalid_request_error",
            code=HTTPStatus.BAD_REQUEST.value,
        ).model_dump(),
        status_code=HTTPStatus.BAD_REQUEST.value,
    )


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
async def create_chat_completion(
    request: ChatCompletionRequest, raw_request: fastapi.Request
):
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

    server_config = ServerConfig.from_args(args)
    log_level = args.log_level

    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level=log_level,
        timeout_keep_alive=server_config.timeout_keep_alive,
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
        "--log-level",
        type=str,
        default="info",
        help="Log level for the engine",
    )
    args = parser.parse_args()

    run_server(args)
