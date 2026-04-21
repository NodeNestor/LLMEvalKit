"""Top-level CLI for LLMEvalKit.

  python -m llmevalkit run --model configs/examples/qwen_finetune.yaml --profile quick
  python -m llmevalkit list-profiles
  python -m llmevalkit list-runners
  python -m llmevalkit list-adapters
"""
from __future__ import annotations

import time
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import adapters as adapters_mod
from . import runners as runners_mod
from .profiles import list_profiles, load_profile
from .results import run_dir, save_run, make_scorecard
from .leaderboard import aggregate as lb_aggregate, render_markdown as lb_render

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="LLMEvalKit — universal LLM eval harness.")
console = Console()


@app.command("list-profiles")
def cmd_list_profiles() -> None:
    t = Table(title="Available profiles")
    t.add_column("name"); t.add_column("description")
    for n in list_profiles():
        p = load_profile(n)
        t.add_row(n, p.get("description", ""))
    console.print(t)


@app.command("list-runners")
def cmd_list_runners() -> None:
    t = Table(title="Available runners")
    t.add_column("name"); t.add_column("requires")
    for name, cls in runners_mod.RUNNERS.items():
        t.add_row(name, ", ".join(cls.requires) or "-")
    console.print(t)


@app.command("list-adapters")
def cmd_list_adapters() -> None:
    t = Table(title="Available adapters")
    t.add_column("kind"); t.add_column("class")
    for k, cls in adapters_mod.ADAPTERS.items():
        t.add_row(k, cls.__name__)
    console.print(t)


@app.command("run")
def cmd_run(
    model: Path = typer.Option(..., help="Path to model YAML config."),
    profile: str = typer.Option(..., help="Profile name or YAML path."),
    results_dir: Path = typer.Option(Path("results"), help="Where to write run output."),
    frameworks_dir: Path = typer.Option(Path("frameworks"), help="Where cloned frameworks live."),
    dry_run: bool = typer.Option(False, help="Parse configs and print plan — don't run."),
) -> None:
    """Run an evaluation profile against a model."""
    model_cfg = yaml.safe_load(model.read_text())
    prof = load_profile(profile)

    console.rule(f"[bold]LLMEvalKit[/] — {model_cfg.get('name', model.stem)} × {prof['name']}")
    console.print(f"Runners: {[r['name'] for r in prof['runners']]}")
    console.print(f"Adapter: {model_cfg['kind']}")

    if dry_run:
        console.print("[yellow]dry-run — exiting[/]")
        return

    adapter_kind = model_cfg.pop("kind")
    adapter_name = model_cfg.get("name", adapter_kind)
    adapter = adapters_mod.make_adapter(adapter_kind, **model_cfg)

    out_root = run_dir(results_dir, adapter_name, prof["name"])
    console.print(f"Writing to: [cyan]{out_root}[/]")

    t_all = time.time()
    all_results = []

    try:
        console.print("[dim]Starting adapter / inference server…[/]")
        endpoint = adapter.start()
        console.print(f"[green]Endpoint live:[/] {endpoint.base_url} (model={endpoint.model_name})")

        for spec in prof["runners"]:
            rname = spec["name"]
            tasks = spec.get("tasks", [])
            args = spec.get("args", {}) or {}
            console.rule(f"[bold cyan]{rname}[/]")
            runner = runners_mod.make_runner(rname, frameworks_dir=frameworks_dir)
            out_sub = out_root / rname.replace("/", "_")
            try:
                result = runner.run(endpoint=endpoint, tasks=tasks, output_dir=out_sub, **args)
            except Exception as e:
                from .runners.base import RunResult
                result = RunResult(runner=rname, tasks=tasks, scores={},
                                   duration_s=0.0, success=False, error=str(e))
            runner.write_result(result, out_sub)
            all_results.append(result)
            status = "[green]PASS[/]" if result.success else "[red]FAIL[/]"
            console.print(f"{status} {rname} — {result.duration_s/60:.1f} min")
    finally:
        console.print("[dim]Stopping adapter…[/]")
        adapter.stop()

    wall = time.time() - t_all
    save_run(out_root, {"name": adapter_name, "kind": adapter_kind, **model_cfg},
             prof, all_results, wall)
    card = make_scorecard(all_results, adapter_name, prof["name"],
                          out_root / "scorecard.md")
    console.rule("[bold green]Done[/]")
    console.print(f"Scorecard: [cyan]{card}[/]")
    console.print(f"Wall time: {wall/60:.1f} min")


@app.command("leaderboard")
def cmd_leaderboard(
    results_dir: Path = typer.Option(Path("results"), help="Where run outputs live."),
    out: Path = typer.Option(None, help="Write to file; default stdout."),
) -> None:
    """Aggregate every run in results/ into LEADERBOARD.md."""
    data, models = lb_aggregate(results_dir)
    md = lb_render(data, models)
    if out:
        out.write_text(md, encoding="utf-8")
        console.print(f"[green]wrote[/] {out}")
    else:
        console.print(md)


if __name__ == "__main__":
    app()
