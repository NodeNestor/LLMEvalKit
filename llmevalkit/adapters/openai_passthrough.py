from __future__ import annotations
from .base import ModelAdapter, Endpoint


class OpenAIPassthrough(ModelAdapter):
    """Use an existing OpenAI-compatible endpoint.

    Covers: OpenAI, Anthropic-via-proxy, Together, Groq, local llama-server
    already running, vLLM server already running, LM Studio, Ollama, SGLang, TGI.
    """

    def __init__(self, name: str, base_url: str, model_name: str,
                 api_key: str = "dummy", extra_headers: dict | None = None, **kwargs):
        super().__init__(name=name)
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"
        self.model_name = model_name
        self.api_key = api_key
        self.extra_headers = extra_headers or {}

    def start(self) -> Endpoint:
        self.wait_for_endpoint(self.base_url, timeout=10.0)
        return Endpoint(
            base_url=self.base_url,
            model_name=self.model_name,
            api_key=self.api_key,
            extra_headers=self.extra_headers,
        )
