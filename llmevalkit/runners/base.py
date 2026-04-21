from __future__ import annotations

import abc
import json
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

from ..adapters.base import Endpoint


@dataclass
class RunResult:
    runner: str
    tasks: list[str]
    scores: dict[str, Any]
    duration_s: float
    success: bool = True
    error: Optional[str] = None
    raw_output_dir: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Runner(abc.ABC):
    """Wraps an external eval framework.

    Each runner takes an OpenAI-compatible Endpoint, runs tasks, returns scores.
    The framework itself may be a Python API (lm-eval) or a subprocess CLI
    (BFCL, RULER, etc.) — subclasses handle that.
    """

    name: str = "base"
    requires: list[str] = []  # cloned frameworks needed under frameworks/

    def __init__(self, frameworks_dir: Path | str = "frameworks", **kwargs):
        self.frameworks_dir = Path(frameworks_dir)
        self.extra = kwargs

    def framework_path(self, repo_name: str) -> Path:
        return self.frameworks_dir / repo_name

    def check_installed(self) -> None:
        for repo in self.requires:
            p = self.framework_path(repo)
            if not p.exists():
                raise RuntimeError(
                    f"Required framework not installed: {p}\n"
                    f"Run: python scripts/install.py --framework {repo}"
                )

    @abc.abstractmethod
    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path, **kwargs) -> RunResult:
        ...

    @staticmethod
    def run_cmd(cmd: list[str], cwd: Optional[Path] = None, env: Optional[dict] = None,
                log_path: Optional[Path] = None, timeout: Optional[float] = None) -> int:
        """Run a subprocess, stream output to log_path if given, return exit code."""
        import os
        import shutil
        proc_env = os.environ.copy()
        if env:
            proc_env.update({k: str(v) for k, v in env.items()})

        # Windows: resolve command name via shutil.which (Popen doesn't do PATHEXT/.exe lookup reliably)
        if cmd:
            resolved = shutil.which(cmd[0])
            if resolved:
                cmd = [resolved, *cmd[1:]]

        stdout = subprocess.PIPE
        log_file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("wb")
            stdout = log_file

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd) if cwd else None, env=proc_env,
                stdout=stdout, stderr=subprocess.STDOUT,
            )
            rc = proc.wait(timeout=timeout)
        finally:
            if log_file:
                log_file.close()
        return rc

    @staticmethod
    def write_result(result: RunResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
