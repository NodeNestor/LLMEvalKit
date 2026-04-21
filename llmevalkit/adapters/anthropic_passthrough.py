from __future__ import annotations

import os
from .base import ModelAdapter, Endpoint


class AnthropicPassthrough(ModelAdapter):
    """Use Anthropic's OpenAI-compatible endpoint.

    Anthropic ships an /v1/chat/completions compatibility layer; point any
    OpenAI-style eval runner at it.
    """

    def __init__(
        self,
        name: str,
        model_name: str = "claude-opus-4-7",
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key=")

    def start(self) -> Endpoint:
        return Endpoint(
            base_url=self.base_url,
            model_name=self.model_name,
            api_key=self.api_key,
            extra_headers={"anthropic-version": "2023-06-01"},
        )
