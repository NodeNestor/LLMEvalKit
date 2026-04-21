"""Unified leaderboard — aggregate every run under results/ into one table.

Maps per-framework metric names to a canonical "score" per (runner, task).
Run directly:
    python -m llmevalkit.leaderboard              # print to stdout
    python -m llmevalkit.leaderboard --out LEADERBOARD.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------
# Canonical metric mapping: (runner, task) -> (metric_path, category, label)
#
# metric_path is a dotted walk through the raw scores dict. For lm-eval:
#     results[task][metric_name]
# For inspect-ai results dump: results.scores[].metrics.<name>.value
# For simple-chat: scores.prompts_ok / prompts_total
#
# If multiple metrics make sense, we pick the "headline" one (flex-extract
# over strict-match because strict frequently reports 0 on small models;
# inst-strict for ifeval; verify.accuracy for code). Alt metrics stay in
# the raw JSON for deeper analysis.
# --------------------------------------------------------------------------

# (runner, task, canonical_label, category, extractor)
# extractor: callable(scores_dict) -> float | None
CATEGORIES = ["math", "code", "instruction", "knowledge", "safety", "chat", "agent", "other"]


def _lm_eval_flex(scores: dict, task: str) -> Optional[float]:
    t = scores.get(task) or {}
    # Try flex-extract first (small-model friendly), then strict
    for k in ("exact_match,flexible-extract", "exact_match,strict-match", "acc,none", "acc_norm,none"):
        if k in t and isinstance(t[k], (int, float)):
            return float(t[k])
    # fallback: first numeric value
    for v in t.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _inspect_score(scorers: list[dict], prefer: list[str]) -> Optional[float]:
    """Pick a metric from inspect results.scores[] in preference order."""
    if not scorers:
        return None
    # Build {scorer_name: {metric_name: value}}
    flat: dict[str, dict[str, float]] = {}
    for s in scorers:
        name = s.get("name", "")
        metrics = s.get("metrics", {}) or {}
        flat[name] = {m: v.get("value") for m, v in metrics.items()
                       if isinstance(v, dict) and "value" in v}
    for full_key in prefer:
        scorer, metric = full_key.split(".", 1)
        v = flat.get(scorer, {}).get(metric)
        if isinstance(v, (int, float)):
            return float(v)
    # Fallback: first accuracy
    for scorer, metrics in flat.items():
        if "accuracy" in metrics and isinstance(metrics["accuracy"], (int, float)):
            return float(metrics["accuracy"])
    return None


# Each entry: (runner, task) -> (category, label, headline_metric_spec, unit)
# For lm-eval: spec = "lmeval:<metric>" (if specific) or "lmeval:flex" for auto
# For inspect-ai: spec = "inspect:<scorer>.<metric>" preference list, pipe-separated
# For simple-chat: spec = "simple"
# For bigcodebench: spec = "bcb" (generation count only, not pass@k)
LEADERBOARD_SPEC: dict[tuple[str, str], tuple[str, str, str]] = {
    ("simple-chat", "chat"):                ("chat",        "chat sanity (6 prompts)",      "simple"),
    ("lm-eval-harness", "gsm8k"):           ("math",        "gsm8k (lm-eval, flex)",        "lmeval:flex"),
    ("lm-eval-harness", "gsm8k_cot"):       ("math",        "gsm8k_cot (lm-eval, flex)",    "lmeval:flex"),
    ("lm-eval-harness", "minerva_math"):    ("math",        "MATH (minerva, lm-eval)",      "lmeval:flex"),
    ("lm-eval-harness", "mmlu"):            ("knowledge",   "MMLU (lm-eval)",               "lmeval:flex"),
    ("lm-eval-harness", "mmlu_pro"):        ("knowledge",   "MMLU-Pro (lm-eval)",           "lmeval:flex"),
    ("lm-eval-harness", "bbh"):             ("knowledge",   "BBH (lm-eval)",                "lmeval:flex"),
    ("lm-eval-harness", "ifeval"):          ("instruction", "IFEval (lm-eval)",             "lmeval:flex"),
    ("lm-eval-harness", "arc_challenge"):   ("knowledge",   "ARC-C (lm-eval)",              "lmeval:flex"),
    ("lm-eval-harness", "truthfulqa_mc2"):  ("knowledge",   "TruthfulQA (lm-eval)",         "lmeval:flex"),
    ("inspect-ai",      "gsm8k"):           ("math",        "gsm8k (inspect, strict)",      "inspect:match.accuracy"),
    ("inspect-ai",      "mgsm"):            ("math",        "MGSM (inspect, multilingual)", "inspect:match.accuracy"),
    ("inspect-ai",      "humaneval"):       ("code",        "HumanEval (inspect)",          "inspect:verify.accuracy"),
    ("inspect-ai",      "mbpp"):            ("code",        "MBPP (inspect)",               "inspect:verify.accuracy"),
    ("inspect-ai",      "ifeval"):          ("instruction", "IFEval inst-strict (inspect)", "inspect:instruction_following.inst_strict_acc"),
    ("inspect-ai",      "simpleqa"):        ("knowledge",   "SimpleQA (inspect)",           "inspect:accuracy.accuracy"),
    ("inspect-ai",      "gpqa"):            ("knowledge",   "GPQA (inspect)",               "inspect:choice.accuracy"),
    ("inspect-ai",      "gaia"):            ("agent",       "GAIA (inspect)",               "inspect:match.accuracy"),
    ("inspect-ai",      "swe_bench"):       ("agent",       "SWE-bench Verified (inspect)", "inspect:match.accuracy"),
    ("inspect-ai",      "niah"):            ("other",       "NIAH long-ctx (inspect)",      "inspect:match.accuracy"),
    ("inspect-ai",      "jailbreakbench"):  ("safety",      "JailbreakBench (inspect)",     "inspect:match.accuracy"),
    ("bigcodebench",    "bigcodebench"):    ("code",        "BigCodeBench (generation)",    "bcb"),
}


@dataclass
class Cell:
    value: Optional[float] = None
    label: str = "-"
    raw: Optional[str] = None


def _load_summary(run_dir: Path) -> Optional[dict]:
    f = run_dir / "summary.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _find_lm_eval_results(run_dir: Path) -> list[dict]:
    """Read every lm-eval results_*.json under the run dir; return list of 'results' maps."""
    out = []
    for jf in run_dir.rglob("results_*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
            if "results" in data:
                out.append(data["results"])
        except Exception:
            pass
    return out


def _inspect_dump(eval_file: Path) -> Optional[dict]:
    import os
    import shutil
    bin_path = shutil.which("inspect") or "inspect"
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    try:
        r = subprocess.run(
            [bin_path, "log", "dump", str(eval_file)],
            capture_output=True, timeout=120, check=False, env=env,
        )
        if r.returncode != 0:
            return None
        return json.loads(r.stdout.decode("utf-8", errors="replace"))
    except Exception:
        return None


def _extract(spec: str, *, lm_eval_map: dict, inspect_scorers: list[dict],
             simple_scores: dict, bcb_samples: int, task: str) -> Optional[float]:
    if spec == "simple":
        total = simple_scores.get("prompts_total", 0)
        ok = simple_scores.get("prompts_ok", 0)
        if total:
            return ok / total
        return None
    if spec == "bcb":
        return float(bcb_samples) if bcb_samples > 0 else None
    if spec.startswith("lmeval:"):
        return _lm_eval_flex(lm_eval_map, task)
    if spec.startswith("inspect:"):
        prefer = spec.removeprefix("inspect:").split("|")
        return _inspect_score(inspect_scorers, prefer)
    return None


def score_run(run_dir: Path) -> dict[tuple[str, str], Cell]:
    """For one run, return {(runner, task) -> Cell}."""
    summary = _load_summary(run_dir)
    if summary is None:
        return {}

    # Build per-runner score map by merging all lm-eval result files
    lm_eval_merged: dict = {}
    for res in _find_lm_eval_results(run_dir):
        lm_eval_merged.update(res)

    # simple-chat: pull from summary.runs[]
    simple_scores = {}
    for r in summary.get("runs", []):
        if r.get("runner") == "simple-chat":
            simple_scores = r.get("scores", {}) or {}
            break

    # bigcodebench: count generated jsonl samples
    bcb_samples = 0
    for jf in (run_dir / "bigcodebench").rglob("*.jsonl") if (run_dir / "bigcodebench").exists() else []:
        bcb_samples += sum(1 for _ in jf.open(encoding="utf-8", errors="replace"))

    # Inspect-ai: load each .eval via `inspect log dump`, cache by task name
    inspect_cache: dict[str, list[dict]] = {}
    insp_dir = run_dir / "inspect-ai"
    if insp_dir.exists():
        for ef in insp_dir.rglob("*.eval"):
            task = ef.parent.name
            data = _inspect_dump(ef)
            if data:
                scorers = data.get("results", {}).get("scores", [])
                # If multiple .eval for same task (e.g. multiple epochs), take latest
                inspect_cache[task] = scorers

    cells: dict[tuple[str, str], Cell] = {}
    for (runner, task), (cat, label, spec) in LEADERBOARD_SPEC.items():
        if runner == "inspect-ai" and task not in inspect_cache:
            continue
        scorers = inspect_cache.get(task, [])
        v = _extract(spec, lm_eval_map=lm_eval_merged, inspect_scorers=scorers,
                     simple_scores=simple_scores, bcb_samples=bcb_samples, task=task)
        if v is not None:
            cells[(runner, task)] = Cell(value=v, label=label)
    return cells


def aggregate(results_root: Path) -> tuple[dict, list[str]]:
    """Scan all runs; keep MOST RECENT run per (model, profile)."""
    latest: dict[tuple[str, str], Path] = {}
    for rd in sorted(results_root.iterdir()):
        if not rd.is_dir():
            continue
        summary = _load_summary(rd)
        if not summary:
            continue
        model = summary.get("model", {}).get("name") or rd.name
        profile = summary.get("profile") or "unknown"
        key = (model, profile)
        if key not in latest or rd.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = rd

    # Build model → scores map (merging across profiles — keep higher score if same task)
    board: dict[str, dict[tuple[str, str], Cell]] = {}
    run_times: dict[str, float] = {}
    for (model, profile), rd in latest.items():
        cells = score_run(rd)
        if not cells:
            continue
        if model not in board:
            board[model] = {}
        for k, cell in cells.items():
            existing = board[model].get(k)
            if existing is None or (cell.value or 0) > (existing.value or 0):
                board[model][k] = cell
        run_times[model] = run_times.get(model, 0) + (_load_summary(rd).get("wall_time_s") or 0)

    models = sorted(board.keys())
    return {"board": board, "run_times": run_times}, models


def render_markdown(data: dict, models: list[str]) -> str:
    board: dict[str, dict[tuple[str, str], Cell]] = data["board"]

    # Collect every (runner, task) cell actually present
    present: dict[tuple[str, str], tuple[str, str]] = {}
    for m in models:
        for k in board.get(m, {}):
            if k in LEADERBOARD_SPEC:
                cat, label, _ = LEADERBOARD_SPEC[k]
                present[k] = (cat, label)

    # Group by category in CATEGORIES order
    by_cat: dict[str, list[tuple[tuple[str, str], str]]] = {c: [] for c in CATEGORIES}
    for k, (cat, label) in present.items():
        by_cat.setdefault(cat, []).append((k, label))

    lines: list[str] = []
    lines.append("# LLMEvalKit Leaderboard")
    lines.append("")
    lines.append("Unified scores across every model evaluated. One canonical metric per task (flex-extract for math, verify.accuracy for code, inst-strict for IFEval, etc.). Generated by `python -m llmevalkit.leaderboard`.")
    lines.append("")
    lines.append(f"**Models:** {len(models)} &nbsp;|&nbsp; **Tasks with scores:** {len(present)}")
    lines.append("")

    # Wall-time row
    lines.append("## Models")
    lines.append("")
    lines.append("| model | total wall-time (min) | tasks covered |")
    lines.append("| --- | ---: | ---: |")
    for m in models:
        wt = data["run_times"].get(m, 0) / 60
        covered = len(board.get(m, {}))
        lines.append(f"| `{m}` | {wt:.1f} | {covered} |")
    lines.append("")

    for cat in CATEGORIES:
        cells = by_cat.get(cat, [])
        if not cells:
            continue
        lines.append(f"## {cat}")
        lines.append("")
        header = "| task | " + " | ".join(f"`{m}`" for m in models) + " |"
        sep = "|" + "|".join(["---"] * (len(models) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for k, label in sorted(cells, key=lambda kv: kv[1]):
            row = [label]
            for m in models:
                c = board.get(m, {}).get(k)
                if c is None or c.value is None:
                    row.append("—")
                else:
                    # For simple-chat show 6/6 style; bcb show sample count; else %
                    runner, task = k
                    if runner == "simple-chat":
                        row.append(f"{int(c.value * 6)}/6")
                    elif runner == "bigcodebench":
                        row.append(f"{int(c.value)} gen")
                    else:
                        row.append(f"{c.value * 100:.1f}%")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("### Notes")
    lines.append("- Scores are the **most recent run per (model, profile)** merged across profiles. If the same task was run twice for a model, the higher score wins.")
    lines.append("- Sample sizes vary — check each run's `scorecard.md` for per-run stderr.")
    lines.append("- BigCodeBench reports **generation count** not pass@k (Windows blocks e2b sandbox eval).")
    lines.append("- Missing cells (—) = the model hasn't been run on that task yet.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Write markdown to this file. Default: stdout.")
    args = ap.parse_args()

    data, models = aggregate(args.results)
    md = render_markdown(data, models)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
