from __future__ import annotations

import shutil
import subprocess
from typing import Optional

from .base import ModelAdapter, Endpoint


class HFvLLM(ModelAdapter):
    """Serve an HF transformers model via vLLM's OpenAI-compatible server.

    Requires: `pip install vllm` (Linux / WSL / Docker on Windows).
    """

    def __init__(
        self,
        name: str,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
        extra_args: Optional[list[str]] = None,
        binary: str = "vllm",
        startup_timeout: float = 600.0,
        **kwargs,
    ):
        super().__init__(name=name)
        self.model = model
        self.host = host
        self.port = port
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.extra_args = extra_args or []
        self.binary = binary
        self.startup_timeout = startup_timeout
        self._proc: Optional[subprocess.Popen] = None

    def _build_cmd(self) -> list[str]:
        bin_path = shutil.which(self.binary) or self.binary
        cmd = [
            bin_path, "serve", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        if self.max_model_len:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.trust_remote_code:
            cmd += ["--trust-remote-code"]
        if self.enforce_eager:
            cmd += ["--enforce-eager"]
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
        return Endpoint(base_url=base_url, model_name=self.model)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
