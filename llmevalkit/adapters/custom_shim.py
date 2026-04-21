from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from .base import ModelAdapter, Endpoint


class CustomShim(ModelAdapter):
    """Spawns a user-supplied FastAPI script that exposes /v1/chat/completions.

    Use this for custom models: finetunes, frankenmerges, novel architectures,
    1-bit/ES-trained weights — anything that doesn't fit a standard inference
    server.

    The shim script must:
      - accept --host, --port CLI args
      - serve /v1/chat/completions and /v1/models (OpenAI-compatible)
      - use scripts/custom_model_server_template.py as a starting point

    Example config:
        kind: custom
        script: /path/to/your/project/serve.py
        model_name: my-custom-v1
        port: 8001
        env:
            CUDA_VISIBLE_DEVICES: "0"
    """

    def __init__(
        self,
        name: str,
        script: str,
        model_name: str,
        host: str = "127.0.0.1",
        port: int = 8001,
        env: Optional[dict] = None,
        extra_args: Optional[list[str]] = None,
        startup_timeout: float = 300.0,
        python: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.script = str(Path(script).resolve())
        self.model_name = model_name
        self.host = host
        self.port = port
        self.env = env or {}
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.python = python or sys.executable
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> Endpoint:
        import os
        proc_env = os.environ.copy()
        proc_env.update({k: str(v) for k, v in self.env.items()})
        cmd = [
            self.python, self.script,
            "--host", self.host,
            "--port", str(self.port),
            *self.extra_args,
        ]
        self._proc = subprocess.Popen(
            cmd, env=proc_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        base_url = f"http://{self.host}:{self.port}/v1"
        try:
            self.wait_for_endpoint(base_url, timeout=self.startup_timeout)
        except TimeoutError:
            self.stop()
            raise
        return Endpoint(base_url=base_url, model_name=self.model_name)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
