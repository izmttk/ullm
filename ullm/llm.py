import asyncio
import queue
import uuid
from typing import AsyncGenerator

from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import EngineConfig
from .core.common import GenerateOutput, SamplingParams
from .core.engine_client import EngineClient


def init_tokenizer(model: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    return tokenizer


class LLM:
    def __init__(self, config: EngineConfig):
        self.engine = EngineClient(config)
        self.tokenizer = init_tokenizer(config.model)

        self.request_states: dict[str, asyncio.Queue[GenerateOutput | None]] = {}
        self.output_processor_task = asyncio.create_task(self.output_processor())

    async def output_processor(self):
        while True:
            try:
                outputs = await asyncio.to_thread(self.engine.get_output, timeout=0.1)
            except queue.Empty:
                continue
            for output in outputs:
                seq_id = output.seq_id
                new_token_id = output.new_token_id
                if seq_id in self.request_states:
                    q = self.request_states[seq_id]
                    token_str = self.detokenize([[new_token_id]])[0]
                    if output.is_finished:
                        q.put_nowait(
                            GenerateOutput(
                                token_str=token_str,
                                is_finished=True,
                                finish_reason=output.finish_reason,
                                num_prompt_tokens=output.num_prompt_tokens,
                                num_generated_tokens=output.num_generated_tokens,
                            )
                        )
                        q.put_nowait(None)  # Sentinel for end of generation
                    else:
                        q.put_nowait(
                            GenerateOutput(
                                token_str=token_str,
                                is_finished=False,
                                finish_reason=None,
                                num_prompt_tokens=0,
                                num_generated_tokens=0,
                            )
                        )

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts)["input_ids"]  # type: ignore

    def detokenize(self, token_ids: list[list[int]]) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    async def generate(
        self,
        prompts: str | list[int],
        params: SamplingParams,
        sequence_id: str | None = None,
    ) -> AsyncGenerator[GenerateOutput, None]:
        if sequence_id is None:
            sequence_id = uuid.uuid4().hex
        try:
            if isinstance(prompts, list):
                token_ids = prompts
            else:
                token_ids = self.tokenize([prompts])[0]
            q: asyncio.Queue[GenerateOutput | None] = asyncio.Queue()

            if params.eos_token_id == -1:
                params.eos_token_id = self.tokenizer.eos_token_id  # type: ignore

            self.request_states[sequence_id] = q
            self.engine.add_sequence(
                sequence_id=sequence_id,
                prompt_token_ids=token_ids,
                sampling_params=params,
            )

            while True:
                output = await q.get()
                if output is None:  # End of generation
                    break
                yield output

            del self.request_states[sequence_id]
        # If the request is disconnected by the client, the
        # generate() task will be canceled. So, we abort the
        # request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            self.abort(sequence_id)
            raise

    def abort(self, sequence_id: str):
        if sequence_id in self.request_states:
            self.engine.abort_sequence(sequence_id)
            del self.request_states[sequence_id]

    async def shutdown(self):
        self.output_processor_task.cancel()
        await asyncio.to_thread(self.engine.shutdown)
