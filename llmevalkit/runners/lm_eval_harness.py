"""EleutherAI lm-evaluation-harness runner.

Supports 250+ tasks: MMLU, MMLU-Pro, BBH, GPQA, IFEval, GSM8K, HellaSwag,
ARC, TruthfulQA, AGIEval, MGSM, XNLI, DROP, many more.

Runs against an OpenAI-compatible endpoint via the `local-chat-completions`
model type, so any adapter works.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class LMEvalHarness(Runner):
    name = "lm-eval-harness"
    requires = ["lm-evaluation-harness"]

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            num_fewshot: int | None = None, limit: int | None = None,
            batch_size: int = 1, apply_chat_template: bool = True,
            timeout: float = 14400.0, **kwargs) -> RunResult:
        self.check_installed()
        repo = self.framework_path("lm-evaluation-harness")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_args = (
            f"base_url={endpoint.base_url}/chat/completions,"
            f"model={endpoint.model_name},"
            f"num_concurrent=8,"
            f"tokenized_requests=False"
        )

        abs_output = output_dir.resolve()
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--model_args", model_args,
            "--tasks", ",".join(tasks),
            "--output_path", str(abs_output),
            "--batch_size", str(batch_size),
        ]
        if num_fewshot is not None:
            cmd += ["--num_fewshot", str(num_fewshot)]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if apply_chat_template:
            cmd += ["--apply_chat_template"]

        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            "OPENAI_BASE_URL": endpoint.base_url,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }

        t0 = time.time()
        log_path = output_dir / "runner.log"
        rc = self.run_cmd(cmd, cwd=repo, env=env, log_path=log_path, timeout=timeout)
        dur = time.time() - t0

        scores: dict = {}
        err: str | None = None
        success = rc == 0
        if success:
            # lm-eval writes results_<timestamp>.json under output_path/<model>/
            for jf in sorted(output_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime):
                try:
                    data = json.loads(jf.read_text())
                    scores = data.get("results", {})
                except Exception:
                    pass
        else:
            err = f"lm-eval-harness exited {rc}; see {log_path}"

        return RunResult(
            runner=self.name, tasks=tasks, scores=scores,
            duration_s=dur, success=success, error=err,
            raw_output_dir=str(output_dir),
        )
