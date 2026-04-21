from .base import ModelAdapter, Endpoint
from .openai_passthrough import OpenAIPassthrough
from .gguf_llamacpp import GGUFLlamaCpp
from .hf_vllm import HFvLLM
from .anthropic_passthrough import AnthropicPassthrough
from .custom_shim import CustomShim

ADAPTERS = {
    "openai": OpenAIPassthrough,
    "gguf": GGUFLlamaCpp,
    "hf": HFvLLM,
    "vllm": HFvLLM,
    "anthropic": AnthropicPassthrough,
    "custom": CustomShim,
}


def make_adapter(kind: str, **kwargs) -> ModelAdapter:
    if kind not in ADAPTERS:
        raise ValueError(f"Unknown adapter kind: {kind}. Available: {list(ADAPTERS)}")
    return ADAPTERS[kind](**kwargs)


__all__ = [
    "ModelAdapter",
    "Endpoint",
    "make_adapter",
    "ADAPTERS",
]
