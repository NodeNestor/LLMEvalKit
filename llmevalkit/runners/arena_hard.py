"""Arena-Hard-Auto runner — LLM-judge proxy for Chatbot Arena.

500 hard queries, judged by a strong model (GPT-4-ish). Cheapest signal
for "how good is this model at chat" without paying for real Arena votes.

USAGE NOTES:
- Supports any OpenAI-compatible endpoint via YAML config (see example in
  arena-hard-auto/config/api_config.yaml: set `endpoints: [{api_base: URL}]`).
- Requires a **judge model API key** — GPT-4o-2024-08-06 is canonical.
  Without a judge, gen_answer.py works but you can't score.
- Runner writes an override.json; for the real workflow, edit
  frameworks/arena-hard-auto/config/api_config.yaml by hand and run
  `gen_answer.py` + `gen_judgment.py` + `show_result.py` manually the first time.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class ArenaHardAuto(Runner):
    name = "arena-hard"
    requires = ["arena-hard-auto"]

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            judge_model: str = "gpt-4o-2024-08-06",
            judge_base_url: str | None = None,
            judge_api_key: str | None = None,
            timeout: float = 43200.0, **kwargs) -> RunResult:
        self.check_installed()
        repo = self.framework_path("arena-hard-auto")
        output_dir.mkdir(parents=True, exist_ok=True)

        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            "OPENAI_BASE_URL": endpoint.base_url,
        }
        if judge_api_key:
            env["JUDGE_API_KEY"] = judge_api_key

        # Arena-Hard is config-driven; write a minimal overrides file
        cfg = {
            "model": endpoint.model_name,
            "api_base": endpoint.base_url,
            "judge": judge_model,
            "judge_api_base": judge_base_url or "https://api.openai.com/v1",
        }
        cfg_path = output_dir / "override.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        cmd = ["python", "gen_answer.py", "--model", endpoint.model_name]
        t0 = time.time()
        log = output_dir / "runner.log"
        rc = self.run_cmd(cmd, cwd=repo, env=env, log_path=log, timeout=timeout)
        dur = time.time() - t0

        return RunResult(
            runner=self.name, tasks=tasks, scores={},
            duration_s=dur, success=(rc == 0),
            error=None if rc == 0 else f"Arena-Hard exited {rc}",
            raw_output_dir=str(output_dir),
            extra={"note": "After answers, run show_result.py inside the repo for win-rate."},
        )
