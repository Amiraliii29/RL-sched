"""Synthetic workload generation: hard periodic tasks plus a soft aperiodic stream.

Periods are drawn from a small semi-harmonic set so the hyperperiod stays bounded; random
periods would make the LCM explode and no configuration would be simulable.
"""

from __future__ import annotations

import numpy as np

from rtsched.config import TICKS_PER_MS, TaskGenConfig
from rtsched.model.task import AperiodicTask, PeriodicTask
from rtsched.model.taskset import TaskSet


def generate_periodic(
    n_cores: int, cfg: TaskGenConfig, rng: np.random.Generator
) -> tuple[PeriodicTask, ...]:
    from rtsched.generation.uunifast import uunifast_discard

    n_tasks = n_cores * cfg.tasks_per_core
    total_u = n_cores * cfg.utilization_per_core
    utils = uunifast_discard(n_tasks, total_u, rng, u_max=1.0)
    periods = [p * TICKS_PER_MS for p in cfg.periods_ms]

    tasks = []
    for tid, u in enumerate(utils):
        period = int(rng.choice(periods))
        wcet = max(1, min(period, int(round(u * period))))
        deadline = max(wcet, int(round(cfg.deadline_ratio * period)))
        tasks.append(PeriodicTask(tid=tid, period=period, deadline=deadline, wcet=wcet))
    return tuple(tasks)

def generate_aperiodic(
    n_cores: int,
    hyperperiod: int,
    cfg: TaskGenConfig,
    rng: np.random.Generator,
) -> tuple[AperiodicTask, ...]:
    """Poisson arrivals sized so the soft stream demands ``soft_load_ratio * n_cores`` capacity."""
    lo, hi = (int(v * TICKS_PER_MS) for v in cfg.soft_exec_ms)
    mean_exec = max(1.0, (lo + hi) / 2.0)
    target_u = cfg.soft_load_ratio * n_cores
    rate = target_u / mean_exec

    max_margin = int(hi * cfg.soft_deadline_factor[1])
    window = max(1, hyperperiod - max_margin)

    tasks: list[AperiodicTask] = []
    t = 0.0
    aid = 0
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= window:
            break
        exec_time = int(rng.integers(lo, hi + 1))
        factor = rng.uniform(*cfg.soft_deadline_factor)
        arrival = int(t)
        tasks.append(
            AperiodicTask(
                aid=aid,
                arrival=arrival,
                exec_time=exec_time,
                deadline=arrival + max(exec_time + 1, int(exec_time * factor)),
                weight=float(rng.uniform(*cfg.soft_weight)),
            )
        )
        aid += 1
    return tuple(tasks)

def generate_taskset(n_cores: int, cfg: TaskGenConfig, seed: int) -> TaskSet:
    """Deterministic end-to-end workload draw for a given seed."""
    rng = np.random.default_rng(seed)
    periodic = generate_periodic(n_cores, cfg, rng)
    hyperperiod = TaskSet(periodic, ()).hyperperiod
    aperiodic = generate_aperiodic(n_cores, hyperperiod, cfg, rng)
    return TaskSet(periodic, aperiodic)
