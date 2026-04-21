"""NVIDIA RULER runner — serious long-context eval.

13 synthetic tasks × configurable context length. Tests real vs. claimed
context length. Supersedes NIAH.

KNOWN ISSUE (Windows): RULER ships as bash scripts wrapping vLLM/TRT-LLM/
NeMo with a hardcoded model registry in `config_models.sh`. Windows requires
WSL. For basic long-context smoke testing cross-platform, use
`inspect_evals/niah` through the `inspect-ai` runner.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class RULER(Runner):
    name = "ruler"
    requires = ["RULER"]

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            max_seq_length: int = 131072, num_samples: int = 100,
            tokenizer: str | None = None, timeout: float = 43200.0, **kwargs) -> RunResult:
        self.check_installed()
        repo = self.framework_path("RULER") / "scripts"
        output_dir.mkdir(parents=True, exist_ok=True)

        # RULER's run.sh expects a model config; simplest path is its
        # OpenAI-compatible client mode.
        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            "OPENAI_API_BASE": endpoint.base_url,
            "MODEL_NAME": endpoint.model_name,
            "TOKENIZER_PATH": tokenizer or endpoint.model_name,
            "MAX_SEQ_LENGTH": str(max_seq_length),
            "NUM_SAMPLES": str(num_samples),
        }
        cmd = ["bash", "run.sh", endpoint.model_name, "synthetic"]

        t0 = time.time()
        log = output_dir / "runner.log"
        rc = self.run_cmd(cmd, cwd=repo, env=env, log_path=log, timeout=timeout)
        dur = time.time() - t0

        scores: dict = {}
        if rc == 0:
            # RULER writes summary.csv per length
            for csv in (repo.parent / "results").rglob("summary.csv"):
                scores[csv.parent.name] = csv.read_text()

        return RunResult(
            runner=self.name, tasks=tasks, scores=scores,
            duration_s=dur, success=(rc == 0),
            error=None if rc == 0 else f"RULER exited {rc}",
            raw_output_dir=str(output_dir),
        )
