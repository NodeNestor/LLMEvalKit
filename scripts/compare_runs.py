#!/usr/bin/env python
"""Build side-by-side comparison markdown from N run directories.

Usage:
    python scripts/compare_runs.py <run_dir_A> <run_dir_B> [<run_dir_C> ...] > compare.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8", errors="replace"))


def inspect_scores(run_dir: Path) -> dict[str, dict]:
    """Call `inspect log dump` on every .eval file; return {task: metrics-dict}."""
    out: dict[str, dict] = {}
    eval_files = list((run_dir / "inspect-ai").rglob("*.eval"))
    for ef in eval_files:
        task = ef.parent.name
        try:
            r = subprocess.run(
                ["inspect", "log", "dump", str(ef)],
                capture_output=True, timeout=120, check=False,
                env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            )
            text = r.stdout.decode("utf-8", errors="replace")
            data = json.loads(text)
            scores_list = data.get("results", {}).get("scores", [])
            flat = {}
            for s in scores_list:
                name = s.get("name", "score")
                for mname, m in s.get("metrics", {}).items():
                    flat[f"{name}.{mname}"] = m.get("value")
            out[task] = flat
        except Exception as e:
            out[task] = {"error": str(e)}
    return out


def lm_eval_scores(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for jf in (run_dir / "lm-eval-harness").rglob("results_*.json"):
        data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
        for task, res in data.get("results", {}).items():
            out[task] = {k: v for k, v in res.items()
                         if isinstance(v, (int, float)) and not k.startswith(" ")}
    return out


def bigcode_samples(run_dir: Path) -> int:
    jsonl = list((run_dir / "bigcodebench").rglob("*.jsonl"))
    if not jsonl:
        return 0
    return sum(1 for _ in jsonl[0].open(encoding="utf-8", errors="replace"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Run dirs under results/")
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    if not all(r.exists() for r in runs):
        missing = [r for r in runs if not r.exists()]
        print(f"Missing: {missing}", file=sys.stderr)
        sys.exit(2)

    rows: list[tuple[str, dict[str, float | str]]] = []
    names: list[str] = []

    for r in runs:
        summary = load_summary(r)
        name = summary.get("model", {}).get("name") or r.name
        names.append(name)

    # simple-chat
    simple = [load_summary(r) for r in runs]
    row: dict[str, float | str] = {}
    for name, s in zip(names, simple):
        for run_entry in s.get("runs", []):
            if run_entry.get("runner") == "simple-chat":
                ok = run_entry.get("scores", {}).get("prompts_ok", "?")
                tot = run_entry.get("scores", {}).get("prompts_total", "?")
                row[name] = f"{ok}/{tot}"
    rows.append(("simple-chat.prompts_ok_total", row))

    # lm-eval-harness
    lm_by = [lm_eval_scores(r) for r in runs]
    all_lm_tasks = sorted({t for d in lm_by for t in d})
    for task in all_lm_tasks:
        all_metrics = sorted({m for d in lm_by for m in d.get(task, {})})
        for metric in all_metrics:
            row = {}
            for name, d in zip(names, lm_by):
                v = d.get(task, {}).get(metric)
                row[name] = f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "-")
            rows.append((f"lm-eval.{task}.{metric}", row))

    # inspect-ai
    insp_by = [inspect_scores(r) for r in runs]
    all_insp_tasks = sorted({t for d in insp_by for t in d})
    for task in all_insp_tasks:
        all_metrics = sorted({m for d in insp_by for m in d.get(task, {}) if m != "error"})
        for metric in all_metrics:
            row = {}
            for name, d in zip(names, insp_by):
                v = d.get(task, {}).get(metric)
                row[name] = f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "-")
            rows.append((f"inspect.{task}.{metric}", row))

    # bigcodebench (generation-only count)
    bc_row = {}
    for name, r in zip(names, runs):
        bc_row[name] = str(bigcode_samples(r))
    rows.append(("bigcodebench.generated_samples", bc_row))

    # wall times
    wt_row = {}
    for name, r in zip(names, runs):
        s = load_summary(r)
        wt_row[name] = f"{s.get('wall_time_s', 0)/60:.1f} min"
    rows.append(("total.wall_time", wt_row))

    # Print markdown
    print(f"# Comparison — {' vs '.join(names)}")
    print()
    header = "| metric | " + " | ".join(names) + " |"
    sep = "|" + "|".join(["---"] * (len(names) + 1)) + "|"
    print(header)
    print(sep)
    for metric, row in rows:
        cells = [row.get(n, "-") for n in names]
        print(f"| `{metric}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
