from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from ..runners.base import RunResult


def run_dir(root: Path, model_name: str, profile_name: str, ts: str | None = None) -> Path:
    ts = ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model_name.replace("/", "_").replace(":", "_")
    d = root / f"{safe_model}__{profile_name}__{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(run_dir_path: Path, model_config: dict, profile: dict,
             results: list[RunResult], wall_time_s: float) -> Path:
    summary = {
        "model": model_config,
        "profile": profile["name"],
        "profile_yaml": profile,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "wall_time_s": wall_time_s,
        "runs": [r.to_dict() for r in results],
    }
    out = run_dir_path / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return out


def load_run(run_dir_path: Path) -> dict:
    return json.loads((run_dir_path / "summary.json").read_text())
