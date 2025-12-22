from .qwen3 import Qwen3Config, Qwen3ForCausalLM

MODEL_REGISTRY = {
    "Qwen3ForCausalLM": (Qwen3ForCausalLM, Qwen3Config),
}
