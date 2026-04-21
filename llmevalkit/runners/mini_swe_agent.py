"""mini-swe-agent runner on SWE-bench Verified / Lite.

100-line agent → hits ~74%+ on SWE-bench Verified. Tests real agentic
coding. Requires Docker for per-task sandboxes.

KNOWN ISSUE: mini-swe-agent v2.x reorganized module paths; the import
path `minisweagent.run.extra.swebench` no longer exists in v2.2. Check
the v2 migration guide. Also, SWE-bench fundamentally requires Docker
per-task sandboxes (~50GB images, hours of runtime for 500 Verified tasks).
Windows with Docker Desktop works; pure Windows does not.

For local-model smoke testing, use the `inspect_evals/swe_bench` task via
the `inspect-ai` runner instead.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class MiniSWEAgent(Runner):
    name = "mini-swe-agent"
    requires = ["mini-swe-agent"]

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            split: str = "verified", subset: int | None = None,
            timeout: float = 86400.0, **kwargs) -> RunResult:
        self.check_installed()
        repo = self.framework_path("mini-swe-agent")
        output_dir.mkdir(parents=True, exist_ok=True)

        # mini-swe-agent uses LiteLLM model strings — prefix with openai/ for OpenAI-compat
        model_str = f"openai/{endpoint.model_name}"

        cmd = [
            "python", "-m", "minisweagent.run.extra.swebench",
            "--model", model_str,
            "--split", split,
            "--output", str(output_dir),
        ]
        if subset:
            cmd += ["--subset", str(subset)]

        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            "OPENAI_BASE_URL": endpoint.base_url,
            "LITELLM_LOG": "INFO",
        }

        t0 = time.time()
        log = output_dir / "runner.log"
        rc = self.run_cmd(cmd, cwd=repo, env=env, log_path=log, timeout=timeout)
        dur = time.time() - t0

        scores: dict = {}
        if rc == 0:
            summary = output_dir / "results.json"
            if summary.exists():
                try:
                    scores = json.loads(summary.read_text())
                except Exception:
                    pass

        return RunResult(
            runner=self.name, tasks=tasks or [split], scores=scores,
            duration_s=dur, success=(rc == 0),
            error=None if rc == 0 else f"mini-swe-agent exited {rc}",
            raw_output_dir=str(output_dir),
        )
