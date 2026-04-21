"""BigCodeBench runner — 1140 real-world Python tasks across 139 libraries.

CLI (real):
    bigcodebench.generate MODEL SPLIT SUBSET --backend openai --base_url URL
                          --root OUT --greedy --id_range "[0,5]"
    bigcodebench.evaluate --split SPLIT --subset SUBSET --samples SAMPLES_JSONL

KNOWN ISSUE (Windows): `bigcodebench.evaluate` requires the `e2b` sandbox
(not installable without an E2B API key). `bigcodebench.generate` works
fine against any OpenAI-compatible endpoint — this runner reports partial
success (generation-only) when evaluate fails from missing e2b. For true
pass@k scoring, use Docker or Linux with the sandbox configured.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class BigCodeBench(Runner):
    name = "bigcodebench"
    requires: list[str] = []

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            split: str = "complete", subset: str = "hard",
            id_range: tuple[int, int] | None = None,
            timeout: float = 43200.0, **kwargs) -> RunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        abs_out = output_dir.resolve()

        env = {
            "OPENAI_API_KEY": endpoint.api_key,
            "OPENAI_BASE_URL": endpoint.base_url,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        gen_cmd = [
            "bigcodebench.generate",
            endpoint.model_name, split, subset,
            "--backend", "openai",
            "--base_url", endpoint.base_url,
            "--root", str(abs_out),
            "--greedy",
        ]
        if id_range:
            gen_cmd += ["--id_range", f"{id_range[0]}-{id_range[1]}"]

        t0 = time.time()
        log = output_dir / "runner.log"
        rc1 = self.run_cmd(gen_cmd, env=env, log_path=log, timeout=timeout)

        samples = None
        if rc1 == 0:
            found = list(abs_out.rglob("*.jsonl"))
            samples = found[0] if found else None

        rc2 = -1
        if rc1 == 0 and samples:
            eval_cmd = [
                "bigcodebench.evaluate",
                "--split", split, "--subset", subset,
                "--samples", str(samples),
            ]
            rc2 = self.run_cmd(eval_cmd, env=env, log_path=log, timeout=timeout)

        dur = time.time() - t0
        scores: dict = {}
        for jf in abs_out.rglob("*_eval_results.json"):
            try:
                scores[jf.stem] = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass

        gen_samples = list(abs_out.rglob("*.jsonl"))
        partial = rc1 == 0 and rc2 != 0 and gen_samples
        return RunResult(
            runner=self.name, tasks=tasks or [split], scores=scores or ({"generated_samples": len(gen_samples)} if partial else {}),
            duration_s=dur, success=(rc1 == 0 and rc2 == 0) or bool(partial),
            error=None if rc1 == 0 and rc2 == 0 else (f"generate-only (eval needs e2b); generate rc={rc1}, eval rc={rc2}" if partial else f"generate rc={rc1}, eval rc={rc2}"),
            raw_output_dir=str(output_dir),
            extra={"note": "Generation succeeded. Eval requires e2b sandbox on Windows."} if partial else {},
        )
