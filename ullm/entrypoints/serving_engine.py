import json
from http import HTTPStatus
from typing import Union

from ..core.common import SamplingParams
from ..llm import LLM
from .protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)


class OpenAIServing:
    def __init__(self, engine: LLM, model_name: str):
        self.engine = engine
        self.model_name = model_name
        self.tokenizer = engine.tokenizer

    async def _generate_full(
        self, prompt: str, sampling_params: SamplingParams, request_id: str
    ):
        text_outputs = ["" for _ in range(sampling_params.n)]
        finish_reason = None
        num_prompt_tokens = 0
        num_generated_tokens = 0
        async for res in self.engine.generate(prompt, sampling_params, request_id):
            for i in range(sampling_params.n):
                token_str = res.token_str
                text_outputs[i] += token_str
            if res.is_finished:
                finish_reason = (
                    res.finish_reason.name.lower() if res.finish_reason else None
                )
                num_prompt_tokens = res.num_prompt_tokens
                num_generated_tokens = res.num_generated_tokens
        return text_outputs, finish_reason, num_prompt_tokens, num_generated_tokens

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> ErrorResponse:
        return ErrorResponse(message=message, type=err_type, code=status_code.value)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message, err_type=err_type, status_code=status_code
            ).model_dump()
        )
        return json_str

    def _extract_sampling_params(
        self, request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> SamplingParams:
        stop_list = []
        if isinstance(request.stop, str):
            stop_list = [request.stop]
        elif isinstance(request.stop, list):
            stop_list = request.stop

        return SamplingParams(
            n=request.n if request.n is not None else 1,
            temperature=request.temperature if request.temperature is not None else 1.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            top_k=request.top_k if request.top_k is not None else -1,
            min_p=request.min_p if request.min_p is not None else 0.0,
            ignore_eos=request.ignore_eos if request.ignore_eos is not None else False,
            max_new_tokens=request.max_tokens,
            stop=stop_list,
        )
