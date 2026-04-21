from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runners.base import RunResult


def _flatten_scores(scores: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(scores, dict):
        for k, v in scores.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                out.extend(_flatten_scores(v, key))
            elif isinstance(v, (int, float, str, bool)) or v is None:
                out.append((key, v))
    return out


def make_scorecard(results: list[RunResult], model_name: str, profile_name: str,
                   out_path: Path) -> Path:
    lines: list[str] = []
    lines.append(f"# Scorecard: {model_name}")
    lines.append(f"\n**Profile:** `{profile_name}`  ")
    total_time = sum(r.duration_s for r in results)
    lines.append(f"**Wall time:** {total_time / 60:.1f} min  ")
    ok = sum(1 for r in results if r.success)
    lines.append(f"**Runners OK:** {ok}/{len(results)}\n")

    for r in results:
        flag = "PASS" if r.success else "FAIL"
        lines.append(f"## [{flag}] {r.runner}  _{r.duration_s / 60:.1f} min_")
        if not r.success:
            lines.append(f"\n> Error: `{r.error}`\n")
            continue
        flat = _flatten_scores(r.scores)
        if not flat:
            lines.append(f"\n_no parsed scores — see raw: {r.raw_output_dir}_\n")
            continue
        lines.append("\n| metric | value |")
        lines.append("| --- | --- |")
        for k, v in flat[:60]:
            if isinstance(v, float):
                v = f"{v:.4f}"
            lines.append(f"| `{k}` | {v} |")
        if len(flat) > 60:
            lines.append(f"| ... | +{len(flat) - 60} more metrics |")
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
