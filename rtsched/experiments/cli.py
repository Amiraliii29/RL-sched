"""Command line entry point.

    python -m rtsched.experiments.cli all            # sweep + every figure
    python -m rtsched.experiments.cli sweep
    python -m rtsched.experiments.cli figures
    python -m rtsched.experiments.cli show 16 0.75   # one configuration, printed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from rtsched.experiments.pipeline import build_config, run_experiment
from rtsched.config import TaskGenConfig
from rtsched.experiments.runner import SweepSpec, aggregate, run_sweep, save
from rtsched.metrics.collect import core_frame, power_trace_frames, taskset_frame
from rtsched.viz import plots

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
RUNS_CSV = RESULTS / "runs.csv"
GRAIN_CSV = RESULTS / "runs_granularity.csv"

TRACE_CONFIGS = ((8, 0.5), (16, 0.75), (32, 0.75))
GRAIN_VALUES = (2, 3, 4)


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total} runs", end="", flush=True)
    if done == total:
        print()


def _policies(with_rl: bool):
    base = (("ga", "slack"), ("ffd", "slack"))
    return base + (("rl", "rl"),) if with_rl else base


def cmd_sweep(args) -> None:
    spec = SweepSpec(seeds=tuple(range(args.seeds)), policies=_policies(getattr(args, "with_rl", False)))
    workers = args.workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    print(f"sweep: {len(list(spec.points()))} runs on {workers} workers")
    frame = run_sweep(spec, workers=workers, progress=_progress)
    save(frame, RUNS_CSV)
    save(aggregate(frame), TABLES / "summary.csv")
    print(f"wrote {RUNS_CSV} and {TABLES / 'summary.csv'}")

    print("granularity study")
    parts = []
    for grain in GRAIN_VALUES:
        grain_spec = SweepSpec(
            seeds=tuple(range(args.seeds)),
            policies=(("ga", "slack"),),
            taskgen=TaskGenConfig(tasks_per_core=grain),
        )
        part = run_sweep(grain_spec, workers=workers, progress=_progress)
        parts.append(part.assign(tasks_per_core=grain))
    save(pd.concat(parts, ignore_index=True), GRAIN_CSV)
    print(f"wrote {GRAIN_CSV}")


def cmd_figures(args) -> None:
    if not RUNS_CSV.exists():
        sys.exit(f"{RUNS_CSV} not found -- run `sweep` first")
    runs = pd.read_csv(RUNS_CSV)

    written = [
        plots.fig_qos_vs_utilization(runs, FIGURES / "01_qos_vs_utilization.png"),
        plots.fig_qos_states(runs, FIGURES / "02_qos_by_state.png"),
        plots.fig_schedulability(runs, FIGURES / "03_schedulability.png"),
        plots.fig_deadline_miss_ratio(runs, FIGURES / "04_deadline_miss_ratio.png"),
        plots.fig_energy(runs, FIGURES / "05_energy.png"),
    ]
    if GRAIN_CSV.exists():
        written.append(
            plots.fig_schedulability_granularity(
                pd.read_csv(GRAIN_CSV), FIGURES / "10_schedulability_granularity.png"
            )
        )
    written += _trace_figures()
    written += _task_table()
    written += _convergence()
    written += _training_curves()
    for path in written:
        print(f"  {path}")


def _trace_figures() -> list[Path]:
    written = []
    for cores, u in TRACE_CONFIGS:
        out = run_experiment(build_config(cores, u, seed=0, record_trace=True))
        tag = f"{cores}c_u{u}"
        written.append(
            plots.fig_core_power(
                core_frame(out.result),
                FIGURES / f"06_core_power_{tag}.png",
                f"Per-core power  ({cores} cores, u={u})",
            )
        )
        core_segments, chip_segments = power_trace_frames(out.result)
        written.append(
            plots.fig_power_trace(
                chip_segments,
                core_segments,
                out.result.tdp_w,
                FIGURES / f"07_power_trace_{tag}.png",
                f"Instantaneous power over one hyperperiod  ({cores} cores, u={u})",
            )
        )
    return written


def _task_table() -> list[Path]:
    out = run_experiment(build_config(16, 0.75, seed=0))
    frame = taskset_frame(out.taskset, out.allocation)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / "taskset_16c_u0.75.csv", index=False)
    return [
        plots.fig_task_table(
            frame, FIGURES / "08_task_table.png", "Generated task set (16 cores, u=0.75)"
        )
    ]


def _convergence() -> list[Path]:
    history = {}
    for cores, u in TRACE_CONFIGS:
        out = run_experiment(build_config(cores, u, seed=0))
        history[f"{cores} cores, u={u}"] = out.allocation.diagnostics.get("history", [])
    return [plots.fig_ga_convergence(history, FIGURES / "09_ga_convergence.png")]


def _training_curves() -> list[Path]:
    logs = {
        name: pd.read_csv(RESULTS / "rl" / f"{name}_training.csv")
        for name in ("mapping", "online")
        if (RESULTS / "rl" / f"{name}_training.csv").exists()
    }
    if not logs:
        return []
    return [plots.fig_training_curves(logs, FIGURES / "11_rl_training.png")]


def cmd_show(args) -> None:
    out = run_experiment(build_config(args.cores, args.utilization, seed=args.seed))
    for key, value in out.summary.items():
        print(f"  {key:<26} {value}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="rtsched")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sweep", help="run the full experiment grid")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--with-rl", action="store_true", help="include the trained phase-2 policies")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("figures", help="render every figure from results/runs.csv")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("all", help="sweep then figures")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--with-rl", action="store_true", help="include the trained phase-2 policies")
    p.set_defaults(func=lambda a: (cmd_sweep(a), cmd_figures(a)))

    p = sub.add_parser("show", help="print the summary of a single configuration")
    p.add_argument("cores", type=int)
    p.add_argument("utilization", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
