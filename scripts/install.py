#!/usr/bin/env python
"""Clone + pip-install the eval frameworks LLMEvalKit runners wrap.

Per-framework — you don't have to install them all. Each framework adds
~100MB–several GB of data + Python deps. Pick what you need:

    python scripts/install.py --framework lm-evaluation-harness
    python scripts/install.py --framework evalchemy
    python scripts/install.py --all           # everything (heavy, hours)
    python scripts/install.py --list

Note: SWE-bench / OpenHands / OSWorld / WebArena additionally require Docker.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMEWORKS_DIR = ROOT / "frameworks"

# Name -> (git URL, branch/None, post-install commands as list-of-list)
FRAMEWORKS: dict[str, dict] = {
    "lm-evaluation-harness": {
        "url": "https://github.com/EleutherAI/lm-evaluation-harness.git",
        "pip": [".[api]"],
    },
    "lighteval": {
        # Clone fails on Windows (test fixtures contain ':' in filenames).
        # PyPI release works fine.
        "pip_package": ["lighteval[litellm]"],
    },
    "evalchemy": {
        "url": "https://github.com/mlfoundations/evalchemy.git",
        "pip": ["."],
    },
    "gorilla": {  # BFCL lives inside gorilla repo
        "url": "https://github.com/ShishirPatil/gorilla.git",
        "subdir": "berkeley-function-call-leaderboard",
        "pip": ["."],
    },
    "mini-swe-agent": {
        "url": "https://github.com/SWE-agent/mini-swe-agent.git",
        "pip": ["."],
    },
    "SWE-bench": {
        "url": "https://github.com/SWE-bench/SWE-bench.git",
        "pip": ["."],
    },
    "RULER": {
        "url": "https://github.com/NVIDIA/RULER.git",
        "pip": ["-r", "requirements.txt"],
    },
    "HarmBench": {
        "url": "https://github.com/centerforaisafety/HarmBench.git",
        "pip": ["-r", "requirements.txt"],
    },
    "arena-hard-auto": {
        "url": "https://github.com/lmarena/arena-hard-auto.git",
        "pip": ["-r", "requirements.txt"],
    },
    "bigcodebench": {
        "url": "https://github.com/bigcode-project/bigcodebench.git",
        "pip": ["."],
    },
    "LiveCodeBench": {
        "url": "https://github.com/LiveCodeBench/LiveCodeBench.git",
        "pip": ["-e", "."],
    },
    # Pip-only (no clone needed):
    "inspect-ai": {
        "pip_package": ["inspect-ai", "inspect-evals"],
    },
    "evalplus": {
        "pip_package": ["evalplus"],
    },
    "ragas": {
        "pip_package": ["ragas"],
    },
    "mteb": {
        "pip_package": ["mteb"],
    },
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def install_one(name: str, no_pip: bool = False) -> None:
    if name not in FRAMEWORKS:
        print(f"Unknown framework: {name}. See --list.", file=sys.stderr)
        sys.exit(2)
    spec = FRAMEWORKS[name]
    FRAMEWORKS_DIR.mkdir(parents=True, exist_ok=True)

    # Pip-only packages (no clone)
    if "pip_package" in spec:
        if not no_pip:
            run([sys.executable, "-m", "pip", "install", "-U", *spec["pip_package"]])
        return

    repo_dir = FRAMEWORKS_DIR / name
    if not repo_dir.exists():
        run(["git", "clone", "--depth", "1", spec["url"], str(repo_dir)])
    else:
        print(f"(already cloned: {repo_dir})")

    if no_pip:
        return
    pip_cwd = repo_dir / spec["subdir"] if "subdir" in spec else repo_dir
    run([sys.executable, "-m", "pip", "install", *spec["pip"]], cwd=pip_cwd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", action="append", default=[],
                    help="Repeatable. Name from --list.")
    ap.add_argument("--all", action="store_true", help="Install everything (heavy).")
    ap.add_argument("--list", action="store_true", help="List available frameworks.")
    ap.add_argument("--no-pip", action="store_true",
                    help="Just clone, skip pip install.")
    args = ap.parse_args()

    if args.list:
        for k, v in FRAMEWORKS.items():
            kind = "pip" if "pip_package" in v else "clone+pip"
            print(f"  {k:32s}  [{kind}]  {v.get('url','')}")
        return

    targets = args.framework
    if args.all:
        targets = list(FRAMEWORKS.keys())
    if not targets:
        ap.error("pass --framework <name> (repeatable), --all, or --list")

    for name in targets:
        print(f"\n=== installing: {name} ===")
        install_one(name, no_pip=args.no_pip)

    print("\nDone.")


if __name__ == "__main__":
    main()
