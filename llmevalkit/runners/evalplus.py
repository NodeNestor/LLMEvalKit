"""EvalPlus runner — HumanEval+ and MBPP+ with 80× test augmentation.

CLI (real signature):
    evalplus.codegen MODEL DATASET --backend openai --base_url URL --root OUT
    evalplus.evaluate DATASET --samples PATH

KNOWN ISSUE (Windows): EvalPlus's OpenAI provider uses `signal.alarm` which is
Unix-only. On Windows you'll see `AttributeError: module 'signal' has no
attribute 'alarm'`. Use WSL or Docker. BigCodeBench / LiveCodeBench are the
canonical replacements on Windows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import Runner, RunResult
from ..adapters.base import Endpoint


class EvalPlus(Runner):
    name = "evalplus"
    requires: list[str] = []

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            dataset: str | None = None, greedy: bool = True,
            id_range: tuple[int, int] | None = None,
            timeout: float = 14400.0, **kwargs) -> RunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets = tasks or [dataset or "humaneval"]
        abs_out = output_dir.resolve()

        scores: dict = {}
        any_failed = False
        t0 = time.time()
        for ds in datasets:
            env = {
                "OPENAI_API_KEY": endpoint.api_key,
                "OPENAI_BASE_URL": endpoint.base_url,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            }
            gen_cmd = [
                "evalplus.codegen",
                endpoint.model_name, ds,
                "--backend", "openai",
                "--base_url", endpoint.base_url,
                "--root", str(abs_out),
            ]
            if greedy:
                gen_cmd += ["--greedy"]
            if id_range:
                gen_cmd += ["--id_range", f"[{id_range[0]},{id_range[1]}]"]

            log = output_dir / f"{ds}.log"
            rc1 = self.run_cmd(gen_cmd, env=env, log_path=log, timeout=timeout)

            # Samples land in <root>/<model>_temp_X.X_*.jsonl
            samples = None
            if rc1 == 0:
                for jl in abs_out.rglob("*.jsonl"):
                    if ds in jl.parent.name or ds in jl.stem:
                        samples = jl
                        break
                if samples is None:
                    found = list(abs_out.rglob("*.jsonl"))
                    samples = found[0] if found else None

            rc2 = -1
            if rc1 == 0 and samples is not None:
                eval_cmd = [
                    "evalplus.evaluate", ds,
                    "--samples", str(samples),
                ]
                rc2 = self.run_cmd(eval_cmd, env=env, log_path=log, timeout=timeout)

            if rc1 != 0 or rc2 != 0:
                any_failed = True

        # EvalPlus writes <samples>_eval_results.json next to the samples file
        for jf in abs_out.rglob("*_eval_results.json"):
            try:
                scores[jf.stem] = json.loads(jf.read_text())
            except Exception:
                pass

        return RunResult(
            runner=self.name, tasks=datasets, scores=scores,
            duration_s=time.time() - t0,
            success=not any_failed and bool(scores),
            error=None if not any_failed else "one or more datasets failed — check .log",
            raw_output_dir=str(output_dir),
        )
