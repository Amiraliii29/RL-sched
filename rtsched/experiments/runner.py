"""Parallel sweep over core counts, utilizations, seeds and policies."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rtsched.config import PlatformConfig, TaskGenConfig
from rtsched.experiments.pipeline import build_config, run_experiment

CORE_COUNTS = (8, 16, 32)
UTILIZATIONS = (0.25, 0.5, 0.75, 0.85, 0.95, 1.0)
"""The four points the project asks for, plus two that expose the schedulability knee."""

POLICIES = (("ga", "slack"), ("ffd", "slack"))


@dataclass(frozen=True)
class SweepSpec:
    core_counts: tuple[int, ...] = CORE_COUNTS
    utilizations: tuple[float, ...] = UTILIZATIONS
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    policies: tuple[tuple[str, str], ...] = POLICIES
    taskgen: TaskGenConfig | None = None
    platform: PlatformConfig | None = None

    def points(self):
        for offline, online in self.policies:
            for cores in self.core_counts:
                for u in self.utilizations:
                    for seed in self.seeds:
                        yield (cores, u, seed, offline, online, self.taskgen, self.platform)


def _run_point(point) -> dict:
    cores, u, seed, offline, online, taskgen, platform = point
    cfg = build_config(cores, u, seed, offline, online, taskgen=taskgen, platform_cfg=platform)
    return run_experiment(cfg).summary


def run_sweep(spec: SweepSpec, workers: int = 0, progress=None) -> pd.DataFrame:
    points = list(spec.points())
    rows: list[dict] = []
    if workers <= 1:
        for i, point in enumerate(points, 1):
            rows.append(_run_point(point))
            if progress:
                progress(i, len(points))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(_run_point, points, chunksize=1), 1):
                rows.append(row)
                if progress:
                    progress(i, len(points))
    return pd.DataFrame(rows)


def save(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged view with the spread, ready to paste into the report."""
    metrics = [
        "schedulable", "hard_miss_ratio", "soft_miss_ratio", "qos_mean", "qos_weighted",
        "acceptance_ratio", "energy_j", "mean_chip_power_w", "peak_chip_power_w",
        "tdp_violation_ratio", "mean_core_level",
    ]
    grouped = frame.groupby(["policy", "n_cores", "u_per_core"])[metrics]
    out = grouped.mean().round(4)
    out.columns = [f"{c}_mean" for c in out.columns]
    spread = grouped.std().round(4)
    spread.columns = [f"{c}_std" for c in spread.columns]
    return pd.concat([out, spread], axis=1).reset_index()
