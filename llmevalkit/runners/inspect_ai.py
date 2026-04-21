"""UK AISI Inspect AI runner.

200+ pre-built evals via `inspect_evals` package: GAIA, SWE-bench, LiveBench,
SimpleQA, HLE, BrowseComp, NIAH, many more. Strong superset wrapper.

CLI pattern:
    inspect eval inspect_evals/<task> --model openai-api/<name>
                 --model-base-url URL --limit N
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class InspectAI(Runner):
    name = "inspect-ai"
    requires: list[str] = []  # pip package

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            limit: int | None = None, max_connections: int = 10,
            timeout: float = 43200.0, **kwargs) -> RunResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Inspect format: openai-api/<service>/<model>; env vars per service
        service = "llmevalkit"
        model_str = f"openai-api/{service}/{endpoint.model_name}"
        service_upper = service.upper().replace("-", "_")
        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            f"{service_upper}_API_KEY": endpoint.api_key,
            f"{service_upper}_BASE_URL": endpoint.base_url,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }

        all_scores: dict = {}
        any_failed = False
        t0 = time.time()
        for task in tasks:
            task_name = f"inspect_evals/{task}" if "/" not in task else task
            log_dir = output_dir / task.replace("/", "_")
            cmd = [
                "inspect", "eval", task_name,
                "--model", model_str,
                "--model-base-url", endpoint.base_url,
                "--log-dir", str(log_dir.resolve()),
                "--max-connections", str(max_connections),
            ]
            if limit:
                cmd += ["--limit", str(limit)]

            log = output_dir / f"{task.replace('/', '_')}.log"
            rc = self.run_cmd(cmd, env=env, log_path=log, timeout=timeout)
            if rc != 0:
                any_failed = True

            # Inspect writes .eval log files per task
            for eval_log in log_dir.rglob("*.eval"):
                all_scores[task] = {"log_file": str(eval_log)}

        # Try to parse summary via `inspect log dump`
        for task in tasks:
            log_dir = output_dir / task.replace("/", "_")
            for ef in log_dir.rglob("*.eval"):
                dump_cmd = ["inspect", "log", "dump", str(ef)]
                dump_log = output_dir / f"{task.replace('/', '_')}.dump.json"
                try:
                    import subprocess as _sp
                    with dump_log.open("wb") as f:
                        _sp.run(dump_cmd, stdout=f, stderr=_sp.STDOUT, timeout=60)
                    data = json.loads(dump_log.read_text(encoding="utf-8", errors="replace"))
                    if "results" in data:
                        all_scores.setdefault(task, {})["results"] = data["results"]
                except Exception:
                    pass

        return RunResult(
            runner=self.name, tasks=tasks, scores=all_scores,
            duration_s=time.time() - t0,
            success=not any_failed,
            error=None if not any_failed else "one or more tasks failed; see .log",
            raw_output_dir=str(output_dir),
            extra={"note": "Use `inspect view start --log-dir <dir>` for the web report."},
        )
