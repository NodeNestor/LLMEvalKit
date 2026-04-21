"""Berkeley Function-Calling Leaderboard runner.

Tests tool use: single-turn, multi-turn, parallel calls, function selection,
executable evaluation, multilingual. Essential for agent finetunes.

NOTE: BFCL's CLI requires model names to be in its fixed registry
(gpt-*, claude-*, etc.). For custom/local endpoints, prefer `inspect-ai`
with `inspect_evals/bfcl` task — it accepts any OpenAI-compatible endpoint.
This runner still works if your model_name exactly matches a BFCL entry
(useful when benchmarking OpenAI/Anthropic/Gemini through a proxy).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class BFCL(Runner):
    name = "bfcl"
    requires = ["gorilla"]

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            test_category: str = "all", num_threads: int = 4,
            timeout: float = 14400.0, **kwargs) -> RunResult:
        self.check_installed()
        repo = self.framework_path("gorilla") / "berkeley-function-call-leaderboard"
        output_dir.mkdir(parents=True, exist_ok=True)

        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            "OPENAI_BASE_URL": endpoint.base_url,
        }

        gen_cmd = [
            "bfcl", "generate",
            "--model", endpoint.model_name,
            "--test-category", test_category,
            "--num-threads", str(num_threads),
            "--result-dir", str(output_dir / "generations"),
        ]
        eval_cmd = [
            "bfcl", "evaluate",
            "--model", endpoint.model_name,
            "--result-dir", str(output_dir / "generations"),
            "--score-dir", str(output_dir / "scores"),
        ]

        t0 = time.time()
        log = output_dir / "runner.log"
        rc1 = self.run_cmd(gen_cmd, cwd=repo, env=env, log_path=log, timeout=timeout)
        rc2 = 0
        if rc1 == 0:
            rc2 = self.run_cmd(eval_cmd, cwd=repo, env=env, log_path=log, timeout=timeout)
        dur = time.time() - t0
        rc = rc1 or rc2

        scores: dict = {}
        if rc == 0:
            for jf in (output_dir / "scores").rglob("*.json"):
                try:
                    scores[jf.stem] = json.loads(jf.read_text())
                except Exception:
                    pass

        return RunResult(
            runner=self.name, tasks=tasks, scores=scores,
            duration_s=dur, success=(rc == 0),
            error=None if rc == 0 else f"BFCL failed (rc={rc})",
            raw_output_dir=str(output_dir),
        )
