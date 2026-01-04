import time
from random import randint, seed
from ullm import LLM, SamplingParams, EngineConfig
import asyncio

async def consume_async_gen(async_gen):
    async for _ in async_gen:
        pass

async def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = "../Qwen3-8B"
    config = EngineConfig(
        model=path,
        gpu_memory_utilization=0.8,
        max_bs=256,
        tp_size=1,
        pp_size=1,
        nccl_port=29500,
        device_ids=[0],
        context_len=4096,
    )
    llm = LLM(config)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    await consume_async_gen(llm.generate("Benchmark: ", SamplingParams(max_tokens=10)))
    print("Start Profiling ...")
    t = time.time()
    
    tasks = []
    for p, s in zip(prompt_token_ids, sampling_params):
        tasks.append(consume_async_gen(llm.generate(p, s)))
    await asyncio.gather(*tasks)
    
    t = (time.time() - t)
    total_tokens = sum(s.max_tokens or 0 for s in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    await llm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())