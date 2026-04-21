"""LiveCodeBench runner — contamination-resistant coding benchmark.

LeetCode/AtCoder/CodeForces problems with timestamps; filter by problems
released after your model's cutoff.

KNOWN ISSUE: LCB's `OpenAIRunner` hardcodes `OpenAI(api_key=os.getenv("OPENAI_KEY"))`
with no base_url override, so it can't be pointed at a custom endpoint
without monkey-patching. On Linux with vllm, pass your HF model via
`--local_model_path`. For custom endpoints cross-platform, use
`inspect_evals/livecodebench` through the `inspect-ai` runner instead.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class LiveCodeBench(Runner):
    name = "livecodebench"
    requires = ["LiveCodeBench"]

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            release_version: str = "release_v4",
            start_date: str | None = None, end_date: str | None = None,
            scenario: str = "codegeneration",
            timeout: float = 43200.0, **kwargs) -> RunResult:
        self.check_installed()
        repo = self.framework_path("LiveCodeBench")
        output_dir.mkdir(parents=True, exist_ok=True)

        env = {"OPENAI_API_KEY": endpoint.api_key, "OPENAI_BASE_URL": endpoint.base_url}
        cmd = [
            "python", "-m", "lcb_runner.runner.main",
            "--model", endpoint.model_name,
            "--scenario", scenario,
            "--release_version", release_version,
            "--output_dir", str(output_dir),
            "--api_provider", "openai",
        ]
        if start_date:
            cmd += ["--start_date", start_date]
        if end_date:
            cmd += ["--end_date", end_date]

        t0 = time.time()
        log = output_dir / "runner.log"
        rc = self.run_cmd(cmd, cwd=repo, env=env, log_path=log, timeout=timeout)
        dur = time.time() - t0

        scores: dict = {}
        for jf in output_dir.rglob("*_eval.json"):
            try:
                scores[jf.stem] = json.loads(jf.read_text())
            except Exception:
                pass

        return RunResult(
            runner=self.name, tasks=tasks, scores=scores,
            duration_s=dur, success=(rc == 0),
            error=None if rc == 0 else f"LiveCodeBench exited {rc}",
            raw_output_dir=str(output_dir),
        )
