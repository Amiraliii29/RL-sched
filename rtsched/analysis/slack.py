"""Slack-stealing accounting for an EDF core.

The available slack at time ``t`` is the longest interval a highest-priority aperiodic job
may occupy without pushing any hard job past its deadline::

    slack(t) = min over future deadlines d of  (d - t) - demand(t, d)

where ``demand(t, d)`` accumulates ``ceil(work / level)`` ticks over the remaining work of
active hard jobs and the full work of hard jobs released in ``[t, d)``, both with deadline
``<= d``.  Charging the ceiling per job rather than once over the sum mirrors exactly what the
engine does -- a job completes on a tick boundary and forfeits the remainder of that tick --
so the budget can never be optimistic by a rounding error.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rtsched.model.task import Job, PeriodicTask


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def available_slack(
    active_jobs: Iterable[Job],
    tasks: Sequence[PeriodicTask],
    next_release: dict[int, int],
    now: int,
    horizon: int,
    level: int,
) -> int:
    """Slack in ticks, clamped to ``[0, horizon - now]``."""
    if horizon <= now:
        return 0

    items: list[tuple[int, int]] = [
        (j.abs_deadline, j.remaining) for j in active_jobs if j.remaining > 0
    ]
    for task in tasks:
        release = next_release.get(task.tid, horizon)
        while release < horizon:
            deadline = release + task.deadline
            if deadline > horizon:
                break
            items.append((deadline, task.work_units))
            release += task.period

    if not items:
        return horizon - now

    items.sort()
    slack = horizon - now
    demand = 0
    for deadline, work in items:
        demand += ceil_div(work, level)
        margin = (deadline - now) - demand
        if margin < slack:
            slack = margin
            if slack <= 0:
                return 0
    return max(0, slack)
