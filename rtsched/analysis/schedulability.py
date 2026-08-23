"""Uniprocessor EDF feasibility tests used to validate a partitioned allocation.

Execution time is the *discrete* cost the simulator charges, ``ceil(work_units / level)``
ticks, rather than the continuous ``C / (level / LEVEL_DEN)``.  Using the continuous form here
would declare cores feasible that the tick-granular engine then misses deadlines on, at
utilizations close to one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rtsched.analysis.slack import ceil_div
from rtsched.model.task import PeriodicTask

EPS = 1e-9


def exec_ticks(task: PeriodicTask, level: int) -> int:
    """Worst-case execution time of ``task`` in ticks when run at ``level``."""
    return ceil_div(task.work_units, level)


def core_utilization(tasks: Iterable[PeriodicTask], level: int) -> float:
    return sum(exec_ticks(t, level) / t.period for t in tasks)


def demand_bound(tasks: Iterable[PeriodicTask], interval: int, level: int) -> float:
    """Baruah's demand bound function at the given core speed."""
    return sum(
        (1 + (interval - t.deadline) // t.period) * exec_ticks(t, level)
        for t in tasks
        if interval >= t.deadline
    )


def edf_feasible(tasks: Sequence[PeriodicTask], level: int) -> bool:
    """Exact EDF feasibility: utilization bound for implicit deadlines, PDA otherwise."""
    if not tasks:
        return True
    u = core_utilization(tasks, level)
    if u > 1.0 + EPS:
        return False
    if all(t.deadline >= t.period for t in tasks):
        return True

    horizon = _pda_horizon(tasks, u)
    checkpoints = sorted(
        {
            t.deadline + k * t.period
            for t in tasks
            for k in range(0, max(0, (horizon - t.deadline) // t.period) + 1)
            if t.deadline + k * t.period <= horizon
        }
    )
    return all(demand_bound(tasks, d, level) <= d + EPS for d in checkpoints)


def _pda_horizon(tasks: Sequence[PeriodicTask], u: float) -> int:
    from math import lcm

    hyperperiod = lcm(*(t.period for t in tasks))
    if u >= 1.0 - EPS:
        return hyperperiod
    slack = sum((t.period - t.deadline) * t.utilization for t in tasks)
    return int(min(hyperperiod, max(t.deadline for t in tasks) + slack / (1 - u)))


def min_feasible_level(tasks: Sequence[PeriodicTask], levels: Sequence[int]) -> int | None:
    """Lowest frequency level at which ``tasks`` remain EDF-feasible, or ``None``."""
    for level in levels:
        if edf_feasible(tasks, level):
            return level
    return None
