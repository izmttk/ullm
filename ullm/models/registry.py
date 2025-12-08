from .qwen3 import Qwen3ForCausalLM, Qwen3Config

MODEL_REGISTRY = {
    "Qwen3ForCausalLM": (Qwen3ForCausalLM, Qwen3Config),
}