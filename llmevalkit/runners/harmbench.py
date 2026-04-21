"""HarmBench runner — standardized red-team eval.

Attack Success Rate against 33 LLMs × 18 methods. Use for safety regression
tracking after RLHF / alignment training.

KNOWN ISSUE (Windows): HarmBench pins vllm>=0.3.0 and spacy==3.7.2 (both
incompatible with Python 3.12 on Windows — vllm needs Linux, spacy 3.7.2
is 3.11-only). It's also HF-model-oriented rather than API-friendly.
On Windows, use `inspect_evals/jailbreakbench` through the `inspect-ai`
runner for a cross-platform safety smoke test.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class HarmBench(Runner):
    name = "harmbench"
    requires = ["HarmBench"]

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            method: str = "DirectRequest", behaviors: str = "harmbench_behaviors_text_all",
            timeout: float = 43200.0, **kwargs) -> RunResult:
        self.check_installed()
        repo = self.framework_path("HarmBench")
        output_dir.mkdir(parents=True, exist_ok=True)

        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            "OPENAI_BASE_URL": endpoint.base_url,
        }
        cmd = [
            "python", "generate_test_cases.py",
            "--method_name", method,
            "--experiment_name", endpoint.model_name,
            "--behaviors_path", f"./data/behavior_datasets/{behaviors}.csv",
            "--save_dir", str(output_dir),
        ]

        t0 = time.time()
        log = output_dir / "runner.log"
        rc = self.run_cmd(cmd, cwd=repo, env=env, log_path=log, timeout=timeout)
        dur = time.time() - t0

        return RunResult(
            runner=self.name, tasks=tasks, scores={},
            duration_s=dur, success=(rc == 0),
            error=None if rc == 0 else f"HarmBench exited {rc}",
            raw_output_dir=str(output_dir),
            extra={"note": "See save_dir/results.csv for ASR breakdown."},
        )
