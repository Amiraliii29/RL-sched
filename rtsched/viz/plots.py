"""Every figure the project asks for, one function each.

All of them take the tidy frames produced by :mod:`rtsched.metrics.collect` and are written to
accept more than one policy, so phase 2 adds its RL curves by passing a longer frame -- no plot
code changes.  Each figure writes the CSV it was drawn from next to the PNG, which doubles as
the table view for the colours that sit below the 3:1 contrast threshold.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rtsched.viz.style import SEQUENTIAL, SERIES, STATUS, finish, series_style, use_style


def _save(fig, path: Path, frame: pd.DataFrame | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    if frame is not None:
        frame.to_csv(path.with_suffix(".csv"), index=False)
    return path


POLICY_ORDER = ("rl+rl", "ga+slack", "ffd+slack")


def _policies(runs: pd.DataFrame) -> list[str]:
    present = runs["policy"].unique().tolist()
    known = [p for p in POLICY_ORDER if p in present]
    return known + sorted(p for p in present if p not in POLICY_ORDER)


def _by_config(runs: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        runs.groupby(["policy", "n_cores", "u_per_core"], as_index=False)[column]
        .mean()
        .sort_values(["policy", "n_cores", "u_per_core"])
    )


def _line_by_cores(ax, frame: pd.DataFrame, column: str) -> None:
    for i, (cores, group) in enumerate(frame.groupby("n_cores")):
        ax.plot(group["u_per_core"], group[column], label=f"{cores} cores", **series_style(i))


def _facet(runs, column, out, title, ylabel, ylim=None):
    """One panel per policy, three core-count series inside each -- never six lines in one axes."""
    use_style()
    policies = _policies(runs)
    frame = _by_config(runs, column)
    fig, axes = plt.subplots(
        1, len(policies), figsize=(5.4 * len(policies), 4.0), sharey=True, squeeze=False
    )
    for ax, policy in zip(axes[0], policies):
        _line_by_cores(ax, frame[frame["policy"] == policy], column)
        finish(ax, policy, "Utilization per core", ylabel if ax is axes[0][0] else "")
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out, frame)


def fig_qos_vs_utilization(runs: pd.DataFrame, out: Path) -> Path:
    """Required: quality of service delivered to soft aperiodic tasks."""
    return _facet(
        runs, "qos_mean", out, "Soft-task quality of service", "Mean QoS", ylim=(-0.03, 1.05)
    )


def fig_qos_states(runs: pd.DataFrame, out: Path) -> Path:
    """Required: system QoS broken down by service state across every configuration."""
    use_style()
    frame = (
        runs.assign(
            on_time=lambda d: d["acceptance_ratio"] - d["soft_miss_ratio"].clip(upper=d["acceptance_ratio"]),
            late=lambda d: d["soft_miss_ratio"].clip(upper=d["acceptance_ratio"]),
            rejected=lambda d: 1.0 - d["acceptance_ratio"],
        )
        .groupby(["policy", "n_cores", "u_per_core"], as_index=False)[["on_time", "late", "rejected"]]
        .mean()
    )
    frame = frame[frame["policy"] == _policies(runs)[0]].reset_index(drop=True)
    frame["state"] = frame["n_cores"].astype(str) + "c / u=" + frame["u_per_core"].astype(str)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(frame))
    bottom = np.zeros(len(frame))
    for key, color, label in (
        ("on_time", STATUS["good"], "served on time"),
        ("late", STATUS["warning"], "served late"),
        ("rejected", STATUS["critical"], "rejected"),
    ):
        ax.bar(x, frame[key], bottom=bottom, color=color, width=0.72, label=label, linewidth=0)
        bottom += frame[key].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels(frame["state"], rotation=45, ha="right")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    finish(
        ax,
        f"Service state of soft aperiodic tasks by system state  ({_policies(runs)[0]})",
        "",
        "Fraction of arrivals",
        legend=False,
    )
    return _save(fig, out, frame)


def fig_schedulability(runs: pd.DataFrame, out: Path) -> Path:
    """Required: fraction of task sets admitted offline and confirmed miss-free in simulation."""
    return _facet(
        runs,
        "schedulable",
        out,
        "Hard task-set schedulability",
        "Schedulable fraction",
        ylim=(-0.03, 1.05),
    )


def fig_schedulability_granularity(runs: pd.DataFrame, out: Path) -> Path:
    """Supplementary: partitioning difficulty is driven by task granularity, not utilization alone.

    With many small tasks a partitioned EDF allocation exists right up to ``u = 1``; with few
    coarse tasks the bin-packing knee appears well before it.
    """
    use_style()
    frame = (
        runs.groupby(["n_cores", "tasks_per_core", "u_per_core"], as_index=False)["schedulable"]
        .mean()
        .sort_values(["n_cores", "tasks_per_core", "u_per_core"])
    )
    core_counts = sorted(frame["n_cores"].unique())
    fig, axes = plt.subplots(
        1, len(core_counts), figsize=(4.6 * len(core_counts), 4.0), sharey=True, squeeze=False
    )
    for ax, cores in zip(axes[0], core_counts):
        panel = frame[frame["n_cores"] == cores]
        for i, (grain, group) in enumerate(panel.groupby("tasks_per_core")):
            ax.plot(
                group["u_per_core"],
                group["schedulable"],
                label=f"{grain} tasks/core",
                **series_style(i),
            )
        ax.set_ylim(-0.03, 1.05)
        finish(ax, f"{cores} cores", "Utilization per core", "Schedulable fraction" if ax is axes[0][0] else "")
    fig.suptitle(
        "Schedulability vs task granularity", x=0.02, ha="left", fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out, frame)


def fig_deadline_miss_ratio(runs: pd.DataFrame, out: Path) -> Path:
    """Required: deadline miss ratio, one row per task class, one column per policy."""
    use_style()
    policies = _policies(runs)
    rows = (("hard_miss_ratio", "Hard periodic tasks"), ("soft_miss_ratio", "Soft aperiodic tasks"))

    fig, axes = plt.subplots(
        2, len(policies), figsize=(5.4 * len(policies), 7.2), sharey=True, sharex=True, squeeze=False
    )
    frames = []
    for row, (column, label) in enumerate(rows):
        frame = _by_config(runs, column)
        frames.append(frame.assign(metric=column))
        for col, policy in enumerate(policies):
            ax = axes[row][col]
            panel = frame[frame["policy"] == policy]
            _line_by_cores(ax, panel, column)
            if row == 0:
                # The hard row is flat at zero by design; state its peak so the scale is honest.
                ax.annotate(
                    f"peak {panel[column].max():.4f}",
                    (0.97, 0.9),
                    xycoords="axes fraction",
                    ha="right",
                    fontsize=9,
                    color="#52514e",
                )
            finish(
                ax,
                f"{label} · {policy}",
                "Utilization per core" if row == 1 else "",
                "Deadline miss ratio" if col == 0 else "",
                legend=row == 0 and col == 0,
            )
    axes[0][0].set_ylim(-0.03, 1.05)
    fig.suptitle("Deadline miss ratio", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out, pd.concat(frames, ignore_index=True))


def fig_energy(runs: pd.DataFrame, out: Path) -> Path:
    return _facet(runs, "energy_j", out, "Energy per hyperperiod", "Energy (J)")


def fig_core_power(cores: pd.DataFrame, out: Path, title: str) -> Path:
    """Required: power consumed by each core."""
    use_style()
    fig, ax = plt.subplots(figsize=(max(6.4, 0.28 * len(cores)), 3.8))
    norm = plt.Normalize(0, max(1e-9, cores["mean_power_w"].max()))
    ax.bar(
        cores["core"],
        cores["mean_power_w"],
        color=plt.get_cmap(SEQUENTIAL)(0.35 + 0.5 * norm(cores["mean_power_w"])),
        width=0.78,
        linewidth=0,
    )
    mean = cores["mean_power_w"].mean()
    ax.axhline(mean, color=SERIES[1], linewidth=1.6, linestyle="--")
    ax.annotate(
        f"mean {mean:.2f} W",
        (len(cores) - 0.5, mean),
        textcoords="offset points",
        xytext=(-4, 5),
        ha="right",
        fontsize=9,
    )
    ax.grid(axis="x", visible=False)
    finish(ax, title, "Core", "Mean power (W)", legend=False)
    return _save(fig, out, cores)


def fig_power_trace(chip: pd.DataFrame, cores: pd.DataFrame, tdp_w: float, out: Path, title: str) -> Path:
    """Required: instantaneous power over the hyperperiod, chip level and per core."""
    use_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(9.5, 6.0), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.35]}
    )

    ms = chip["start"] / 1000.0
    axes[0].step(ms, chip["power_w"], where="post", color=SERIES[0], linewidth=1.6)
    axes[0].axhline(tdp_w, color=STATUS["critical"], linewidth=1.6, linestyle="--")
    axes[0].annotate(
        f"TDP {tdp_w:.1f} W",
        (ms.iloc[-1], tdp_w),
        textcoords="offset points",
        xytext=(-4, 4),
        ha="right",
        fontsize=9,
        color=STATUS["critical"],
    )
    finish(axes[0], title, "", "Chip power (W)", legend=False)

    grid, extent = _core_power_grid(cores)
    image = axes[1].imshow(
        grid, aspect="auto", origin="lower", cmap=SEQUENTIAL, extent=extent, interpolation="nearest"
    )
    axes[1].grid(visible=False)
    finish(axes[1], "", "Time (ms)", "Core", legend=False)
    fig.colorbar(image, ax=axes[1], pad=0.01, label="Core power (W)")
    return _save(fig, out, chip)


def _core_power_grid(cores: pd.DataFrame, n_bins: int = 600):
    n_cores = int(cores["core"].max()) + 1
    end = int(cores["end"].max())
    edges = np.linspace(0, end, n_bins + 1)
    grid = np.zeros((n_cores, n_bins))
    centres = 0.5 * (edges[:-1] + edges[1:])
    for core, group in cores.groupby("core"):
        index = np.searchsorted(group["start"].to_numpy(), centres, side="right") - 1
        grid[int(core)] = group["power_w"].to_numpy()[np.clip(index, 0, len(group) - 1)]
    return grid, (0.0, end / 1000.0, -0.5, n_cores - 0.5)


def fig_task_table(tasks: pd.DataFrame, out: Path, title: str, max_rows: int = 22) -> Path:
    """Required: table of tasks and their characteristics (full set written to CSV)."""
    use_style()
    view = tasks.head(max_rows).copy()
    for column in ("period_ticks", "deadline_ticks", "wcet_ticks"):
        view[column] = pd.to_numeric(view[column], errors="coerce").div(1000).round(2)
    view = view.rename(
        columns={
            "period_ticks": "T (ms)",
            "deadline_ticks": "D (ms)",
            "wcet_ticks": "C (ms)",
            "utilization": "U",
            "core_level": "freq %",
        }
    )
    view["U"] = pd.to_numeric(view["U"], errors="coerce").round(4)

    fig, ax = plt.subplots(figsize=(9.0, 0.26 * len(view) + 0.8))
    ax.axis("off")
    table = ax.table(
        cellText=view.fillna("-").astype(str).to_numpy(),
        colLabels=view.columns,
        bbox=(0, 0, 1, 1),
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#dcdbd6")
        if row == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title(f"{title}  (first {len(view)} of {len(tasks)} rows; full set in CSV)", loc="left")
    return _save(fig, out, tasks)


def fig_training_curves(logs: dict[str, pd.DataFrame], out: Path) -> Path:
    """Phase 2: PPO learning curves, one column per agent."""
    use_style()
    fig, axes = plt.subplots(1, len(logs), figsize=(5.4 * len(logs), 4.0), squeeze=False)
    frames = []
    for ax, (name, frame) in zip(axes[0], logs.items()):
        frames.append(frame.assign(agent=name))
        smooth = frame["episode_return"].rolling(15, min_periods=1).mean()
        ax.plot(frame["steps"] / 1e6, frame["episode_return"], color=SERIES[0], alpha=0.25, marker="")
        ax.plot(frame["steps"] / 1e6, smooth, color=SERIES[0], marker="")
        finish(ax, f"{name} agent", "Environment steps (millions)", "Episode return", legend=False)
    fig.suptitle("PPO training", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out, pd.concat(frames, ignore_index=True))


def fig_ga_convergence(history: dict[str, list[float]], out: Path) -> Path:
    use_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rows = []
    for i, (label, values) in enumerate(history.items()):
        ax.plot(range(len(values)), values, label=label, marker="", **{"color": SERIES[i % 3]})
        rows += [{"config": label, "generation": g, "best_cost": v} for g, v in enumerate(values)]
    finish(ax, "Genetic-algorithm convergence", "Generation", "Best cost")
    return _save(fig, out, pd.DataFrame(rows))
