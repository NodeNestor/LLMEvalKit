from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class Endpoint:
    """Everything a runner needs to talk to a model."""
    base_url: str
    model_name: str
    api_key: str = "dummy"
    extra_headers: dict = None

    def as_env(self) -> dict:
        env = {
            "OPENAI_BASE_URL": self.base_url,
            "OPENAI_API_KEY": self.api_key,
            "OPENAI_MODEL": self.model_name,
        }
        return env


class ModelAdapter(abc.ABC):
    """Turns any model into an OpenAI-compatible endpoint.

    Subclass pattern:
      - __init__ stores config
      - start() spawns whatever inference server and returns Endpoint
      - stop() cleans up subprocesses
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._endpoint: Optional[Endpoint] = None

    @abc.abstractmethod
    def start(self) -> Endpoint:
        """Block until a /v1/chat/completions endpoint is ready."""
        ...

    def stop(self) -> None:
        """Override if your adapter owns subprocesses."""
        return None

    def __enter__(self) -> Endpoint:
        self._endpoint = self.start()
        return self._endpoint

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @staticmethod
    def wait_for_endpoint(base_url: str, timeout: float = 300.0, interval: float = 2.0) -> None:
        """Poll /v1/models until the server is live or timeout."""
        url = base_url.rstrip("/") + "/models"
        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                r = httpx.get(url, timeout=5.0)
                if r.status_code < 500:
                    return
            except Exception as e:
                last_err = e
            time.sleep(interval)
        raise TimeoutError(f"Endpoint not ready after {timeout}s: {url} (last err: {last_err})")
