"""HuggingFace lighteval runner.

Supports 1000+ tasks including HLE, SimpleQA, MMLU-Pro, IFEval. Uses LiteLLM
under the hood so any OpenAI-compatible endpoint works.

KNOWN ISSUE (Windows): lighteval writes cache/output paths containing '|'
(e.g. `.../gsm8k|0`) which the Windows filesystem rejects. This fails every
task out-of-the-box on Windows. Use WSL or Docker. On Linux this runner works
out of the box.

CLI:
    lighteval endpoint litellm "model_name=X,base_url=Y,provider=openai" tasks
            --output-dir OUT --max-samples N
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class LightEval(Runner):
    name = "lighteval"
    requires: list[str] = []  # pip-installed

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            max_samples: int | None = None, timeout: float = 14400.0, **kwargs) -> RunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        abs_out = output_dir.resolve()

        task_arg = ",".join(tasks)
        model_arg = (
            f"model_name=openai/{endpoint.model_name},"
            f"base_url={endpoint.base_url},"
            f"provider=openai"
        )

        cmd = [
            "lighteval", "endpoint", "litellm",
            model_arg, task_arg,
            "--output-dir", str(abs_out),
        ]
        if max_samples:
            cmd += ["--max-samples", str(max_samples)]

        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            "OPENAI_BASE_URL": endpoint.base_url,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }

        t0 = time.time()
        log_path = output_dir / "runner.log"
        rc = self.run_cmd(cmd, env=env, log_path=log_path, timeout=timeout)
        dur = time.time() - t0

        scores: dict = {}
        success = rc == 0
        if success:
            for jf in abs_out.rglob("results_*.json"):
                try:
                    data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
                    scores = data.get("results", {})
                    break
                except Exception:
                    pass

        return RunResult(
            runner=self.name, tasks=tasks, scores=scores,
            duration_s=dur, success=success,
            error=None if success else f"lighteval exited {rc}",
            raw_output_dir=str(output_dir),
        )
