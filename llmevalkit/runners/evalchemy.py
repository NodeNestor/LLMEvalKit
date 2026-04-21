"""Evalchemy runner — wraps lm-eval-harness + post-training benchmarks.

Covers AIME24/25, MATH500, LiveCodeBench, BigCodeBench, GPQA-Diamond,
HumanEval+, MBPP+, MultiPL-E, CRUXEval with proper vLLM data-parallelism.
Most relevant single framework for evaluating coding/math/reasoning finetunes.

KNOWN ISSUE (Windows): evalchemy depends on a custom lm-evaluation-harness
fork that fails to build due to Windows MAX_PATH (arabic_leaderboard task
filenames >260 chars) and vllm (path + '[]' chars). Use WSL or Docker.
Many of the same tasks (AIME, MATH500, GPQA-Diamond) are also in upstream
lm-evaluation-harness — prefer that runner on Windows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class Evalchemy(Runner):
    name = "evalchemy"
    requires = ["evalchemy"]

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            batch_size: int = "auto", timeout: float = 14400.0, **kwargs) -> RunResult:
        self.check_installed()
        repo = self.framework_path("evalchemy")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_args = (
            f"base_url={endpoint.base_url}/chat/completions,"
            f"model={endpoint.model_name},"
            f"num_concurrent=16,"
            f"tokenized_requests=False"
        )

        cmd = [
            "python", "-m", "eval.eval",
            "--model", "openai-chat-completions",
            "--model_args", model_args,
            "--tasks", ",".join(tasks),
            "--output_path", str(output_dir),
            "--batch_size", str(batch_size),
            "--apply_chat_template",
        ]
        env = {"OPENAI_API_KEY": endpoint.api_key, "OPENAI_BASE_URL": endpoint.base_url}

        t0 = time.time()
        log_path = output_dir / "runner.log"
        rc = self.run_cmd(cmd, cwd=repo, env=env, log_path=log_path, timeout=timeout)
        dur = time.time() - t0

        scores: dict = {}
        if rc == 0:
            for jf in sorted(output_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime):
                try:
                    scores = json.loads(jf.read_text()).get("results", {})
                except Exception:
                    pass

        return RunResult(
            runner=self.name, tasks=tasks, scores=scores,
            duration_s=dur, success=(rc == 0),
            error=None if rc == 0 else f"evalchemy exited {rc}",
            raw_output_dir=str(output_dir),
        )
