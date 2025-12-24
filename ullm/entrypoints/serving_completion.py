import time
from http import HTTPStatus
from typing import AsyncGenerator

from fastapi.responses import JSONResponse, StreamingResponse

from .protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    UsageInfo,
)
from .serving_engine import OpenAIServing


class OpenAIServingCompletion(OpenAIServing):
    async def create_completion(self, request: CompletionRequest):
        try:
            if request.echo:
                raise ValueError("echo is not supported")
            if request.suffix:
                raise ValueError("suffix is not supported")
            if request.logprobs is not None and request.logprobs > 0:
                raise ValueError("logprobs is not supported")
            if request.best_of is not None and request.best_of > 1:
                raise ValueError("best_of is not supported")
            if request.logit_bias:
                raise ValueError("logit_bias is not supported")
            if request.presence_penalty is not None and request.presence_penalty != 0.0:
                raise ValueError("presence_penalty is not supported")
            if (
                request.frequency_penalty is not None
                and request.frequency_penalty != 0.0
            ):
                raise ValueError("frequency_penalty is not supported")

            create_time_ns = time.time_ns()
            create_time_sec = create_time_ns // 1_000_000_000

            request_id = f"cmpl-{create_time_ns}"

            if isinstance(request.prompt, list):
                if len(request.prompt) > 1:
                    raise ValueError("Batching is not supported")
                prompt = request.prompt[0]
            else:
                prompt = request.prompt

            sampling_params = self._extract_sampling_params(request)

            if request.stream:
                return StreamingResponse(
                    self.completion_stream_generator(
                        request, prompt, request_id, create_time_sec
                    ),
                    media_type="text/event-stream",
                )
            (
                text_outputs,
                finish_reason,
                num_prompt_tokens,
                num_generated_tokens,
            ) = await self._generate_full(prompt, sampling_params, request_id)
            assert finish_reason == "stop" or finish_reason == "length"

            choices = [
                CompletionResponseChoice(
                    index=i,
                    text=text_outputs[i],
                    logprobs=None,
                    finish_reason=finish_reason,
                )
                for i in range(sampling_params.n)
            ]
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            return CompletionResponse(
                id=request_id,
                created=create_time_sec,
                model=self.model_name,
                choices=choices,
                usage=usage,
            )
        except ValueError as e:
            return JSONResponse(
                content=self.create_error_response(
                    message=str(e),
                    err_type="BadRequest",
                    status_code=HTTPStatus.BAD_REQUEST,
                ).model_dump(),
                status_code=HTTPStatus.BAD_REQUEST.value,
            )
        except Exception as e:
            return JSONResponse(
                content=self.create_error_response(
                    message=f"Internal server error: {str(e)}",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                ).model_dump(),
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            )

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        prompt: str,
        request_id: str,
        created: int,
    ) -> AsyncGenerator[str, None]:
        try:
            sampling_params = self._extract_sampling_params(request)
            for i in range(sampling_params.n):
                choice_data = CompletionResponseStreamChoice(
                    index=i, text="", logprobs=None, finish_reason=None
                )
                chunk = CompletionStreamResponse(
                    id=request_id,
                    object="text_completion",
                    created=created,
                    choices=[choice_data],
                    model=self.model_name,
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

            finish_reason = None
            async for output in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                for i in range(sampling_params.n):
                    text = output.token_str
                    if output.is_finished:
                        finish_reason = output.finish_reason
                    choice_data = CompletionResponseStreamChoice(
                        index=i, text=text, logprobs=None, finish_reason=None
                    )
                    chunk = CompletionStreamResponse(
                        id=request_id,
                        object="text_completion",
                        created=created,
                        choices=[choice_data],
                        model=self.model_name,
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            assert finish_reason is not None
            finish_reason = finish_reason.name.lower()
            assert finish_reason == "stop" or finish_reason == "length"

            for i in range(sampling_params.n):
                choice_data = CompletionResponseStreamChoice(
                    index=i, text="", logprobs=None, finish_reason=finish_reason
                )
                chunk = CompletionStreamResponse(
                    id=request_id,
                    object="text_completion",
                    created=created,
                    choices=[choice_data],
                    model=self.model_name,
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
        except Exception as e:
            data = self.create_streaming_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
