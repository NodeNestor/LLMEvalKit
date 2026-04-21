from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .base import ModelAdapter, Endpoint


class GGUFLlamaCpp(ModelAdapter):
    """Serve a GGUF model via llama-server.

    Requires llama.cpp's `llama-server` on PATH (or pass binary=).

    Windows tip: grab the latest llama-b*-bin-win-cuda-*.zip from
    github.com/ggerganov/llama.cpp/releases and add to PATH.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        model_name: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8787,
        ctx_size: int = 32768,
        n_gpu_layers: int = 999,
        flash_attn: bool = True,
        chat_template: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
        binary: str = "llama-server",
        startup_timeout: float = 180.0,
        **kwargs,
    ):
        super().__init__(name=name)
        self.model_path = str(Path(model_path).resolve())
        self.model_name = model_name or Path(model_path).stem
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.flash_attn = flash_attn
        self.chat_template = chat_template
        self.extra_args = extra_args or []
        self.binary = binary
        self.startup_timeout = startup_timeout
        self._proc: Optional[subprocess.Popen] = None

    def _build_cmd(self) -> list[str]:
        bin_path = shutil.which(self.binary) or self.binary
        cmd = [
            bin_path,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--alias", self.model_name,
        ]
        if self.flash_attn:
            cmd += ["--flash-attn", "on"]
        if self.chat_template:
            cmd += ["--chat-template-file", str(self.chat_template)]
        cmd += self.extra_args
        return cmd

    def start(self) -> Endpoint:
        cmd = self._build_cmd()
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
