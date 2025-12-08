import asyncio
import time
from typing import AsyncGenerator

from fastapi.responses import StreamingResponse

from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from .serving_engine import OpenAIServing


class OpenAIServingChat(OpenAIServing):

    async def create_chat_completion(self, request: ChatCompletionRequest):
        if request.logit_bias:
            return self.create_error_response(400, "logit_bias is not supported")
        if isinstance(request.messages, str):
            return self.create_error_response(400, "string messages are not supported")
        if request.presence_penalty is not None and request.presence_penalty != 0.0:
            return self.create_error_response(400, "presence_penalty is not supported")
        if request.frequency_penalty is not None and request.frequency_penalty != 0.0:
            return self.create_error_response(400, "frequency_penalty is not supported")

        create_time_ns = time.time_ns()
        create_time_sec = create_time_ns // 1_000_000_000

        conversation = [message.model_dump() for message in request.messages]
        prompt_or_tokens = self.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=False, add_generation_prompt=True
        )
        prompt = str(prompt_or_tokens)

        request_id = f"chatcmpl-{create_time_ns}"

        sampling_params = self._extract_sampling_params(request)

        if request.stream:
            return StreamingResponse(
                self.chat_completion_stream_generator(
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
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=text_outputs[i]),
                finish_reason=finish_reason,
            )
            for i in range(sampling_params.n)
        ]
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=create_time_sec,
            model=self.model_name,
            choices=choices,
            usage=usage,
        )

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        prompt: str,
        request_id: str,
        created: int,
    ) -> AsyncGenerator[str, None]:
        sampling_params = self._extract_sampling_params(request)
        for i in range(sampling_params.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role="assistant"), logprobs=None, finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object="chat.completion.chunk",
                choices=[choice_data],
                model=self.model_name,
                created=created
            )
            data = chunk.model_dump_json(exclude_unset=True,)
            yield f"data: {data}\n\n"
        finish_reason = None
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            for i in range(sampling_params.n):
                delta_text = output.token_str
                if output.is_finished:
                    finish_reason = output.finish_reason
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(content=delta_text),
                    logprobs=None,
                    finish_reason=None,
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object="chat.completion.chunk",
                    choices=[choice_data],
                    model=self.model_name,
                    created=created
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
                
        assert finish_reason is not None
        finish_reason = finish_reason.name.lower()
        assert finish_reason == "stop" or finish_reason == "length"

        for i in range(sampling_params.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(), logprobs=None, finish_reason=finish_reason
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object="chat.completion.chunk",
                choices=[choice_data],
                model=self.model_name,
                created=created
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
