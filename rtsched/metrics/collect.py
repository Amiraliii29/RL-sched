"""Derive every reported metric from the job records and power totals of a run."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from rtsched.policies.base import Allocation
from rtsched.sim.trace import SimResult


def summarize(result: SimResult, allocation: Allocation) -> dict:
    hard, soft = result.hard_jobs, result.soft_jobs
    qos = np.array([j.qos for j in soft]) if soft else np.zeros(0)
    weights = np.array([j.weight for j in soft]) if soft else np.zeros(0)
    accepted = [j for j in soft if j.accepted]

    return {
        "offline_feasible": allocation.feasible,
        "n_hard_jobs": len(hard),
        "n_soft_jobs": len(soft),
        "hard_miss_ratio": _ratio([j.missed for j in hard]),
        "soft_miss_ratio": _ratio([j.missed for j in soft]),
        "deadline_miss_ratio": _ratio([j.missed for j in hard + soft]),
        "acceptance_ratio": _ratio([j.accepted for j in soft]),
        "qos_mean": float(qos.mean()) if qos.size else 1.0,
        "qos_weighted": float((qos * weights).sum() / weights.sum()) if weights.sum() else 1.0,
        "qos_accepted": float(np.mean([j.qos for j in accepted])) if accepted else 1.0,
        "soft_response_mean_ticks": _mean(
            [j.finish - j.release for j in accepted if j.finish >= 0]
        ),
        "schedulable": bool(allocation.feasible and not any(j.missed for j in hard)),
        "energy_j": result.energy_j,
        "mean_chip_power_w": result.mean_chip_power_w,
        "peak_chip_power_w": result.peak_chip_power_w,
        "tdp_w": result.tdp_w,
        "tdp_violation_ratio": result.tdp_violation_ratio,
        "mean_core_power_w": float(np.mean(result.core_energy_j)) / (result.horizon * 1e-6),
        "mean_core_busy_ratio": float(np.mean(result.core_busy_ratio)),
        "mean_core_level": float(np.mean(result.core_levels)),
        "sim_steps": result.steps,
        "horizon_ticks": result.horizon,
    }


def jobs_frame(result: SimResult) -> pd.DataFrame:
    return pd.DataFrame([asdict(j) for j in result.jobs])


def core_frame(result: SimResult) -> pd.DataFrame:
    duration_s = result.horizon * 1e-6
    return pd.DataFrame(
        {
            "core": range(len(result.core_energy_j)),
            "energy_j": result.core_energy_j,
            "mean_power_w": [e / duration_s for e in result.core_energy_j],
            "busy_ratio": result.core_busy_ratio,
            "level": result.core_levels,
        }
    )


def power_trace_frames(result: SimResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    if result.trace is None:
        return pd.DataFrame(), pd.DataFrame()
    cores = pd.DataFrame(
        result.trace.core_segments, columns=["core", "start", "end", "level", "power_w"]
    )
    chip = pd.DataFrame(result.trace.chip_segments, columns=["start", "end", "power_w"])
    return cores, chip


def taskset_frame(taskset, allocation: Allocation) -> pd.DataFrame:
    rows = [
        {
            "kind": "hard",
            "id": t.tid,
            "period_ticks": t.period,
            "deadline_ticks": t.deadline,
            "wcet_ticks": t.wcet,
            "utilization": t.utilization,
            "core": allocation.core_of_task.get(t.tid, -1),
            "core_level": allocation.level_of_core[allocation.core_of_task.get(t.tid, 0)],
            "weight": 1.0,
        }
        for t in taskset.periodic
    ]
    rows += [
        {
            "kind": "soft",
            "id": a.aid,
            "period_ticks": pd.NA,
            "deadline_ticks": a.relative_deadline,
            "wcet_ticks": a.exec_time,
            "utilization": pd.NA,
            "core": pd.NA,
            "core_level": pd.NA,
            "weight": a.weight,
        }
        for a in taskset.aperiodic
    ]
    return pd.DataFrame(rows)


def _ratio(flags) -> float:
    return float(np.mean(flags)) if len(flags) else 0.0


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else 0.0
