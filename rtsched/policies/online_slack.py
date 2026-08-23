"""Slack-aware dispatch of soft aperiodic jobs -- the phase-1 online baseline.

For every core the policy walks frequency levels upward from the current one and takes the
lowest level at which the job's whole backlog fits inside the core's stolen slack and still
meets the soft deadline; among those candidates it prefers the one with the most headroom and
the least added power.  If no core can meet the deadline it falls back to the best-effort core
so the job still earns partial QoS credit.
"""

from __future__ import annotations

from dataclasses import dataclass

from rtsched.analysis.slack import ceil_div
from rtsched.config import LEVEL_DEN
from rtsched.model.platform import Platform
from rtsched.model.task import Job
from rtsched.model.taskset import TaskSet
from rtsched.policies.base import (
    Allocation,
    CoreObservation,
    OnlineDecision,
    OnlineObservation,
)


@dataclass
class SlackStealingPolicy:
    w_slack: float = 1.0
    w_backlog: float = 1.0
    w_power: float = 0.5
    allow_boost: bool = True
    best_effort: bool = True

    name: str = "slack"

    def reset(self, taskset: TaskSet, platform: Platform, allocation: Allocation) -> None:
        self._levels = platform.levels

    def on_arrival(self, job: Job, obs: OnlineObservation) -> OnlineDecision:
        work = obs.job_work_ticks
        best: tuple | None = None
        fallback: tuple | None = None

        for core in obs.cores:
            for level in self._candidates(core):
                needed = ceil_div((core.soft_backlog_ticks + work) * LEVEL_DEN, level)
                fits_slack = core.slack_ticks >= needed
                meets_deadline = obs.now + needed <= obs.job_deadline
                cand = (self._score(core, level, needed, work), -level, core.index, level)

                if fits_slack and meets_deadline:
                    if best is None or cand > best:
                        best = cand
                    break
                if fallback is None or cand > fallback:
                    fallback = cand

        if best is not None:
            return OnlineDecision(True, best[2], best[3])
        if self.best_effort and fallback is not None:
            return OnlineDecision(True, fallback[2], fallback[3])
        return OnlineDecision.reject()

    def _candidates(self, core: CoreObservation):
        if not self.allow_boost:
            return (core.level,)
        return tuple(l for l in self._levels if l >= core.level) or (core.level,)

    def _score(self, core: CoreObservation, level: int, needed: int, work: int) -> float:
        headroom = min(1.0, core.slack_ticks / max(1, needed))
        backlog = core.soft_backlog_ticks / max(1, work)
        span = max(1, core.max_level - core.base_level)
        return (
            self.w_slack * headroom
            - self.w_backlog * backlog
            - self.w_power * (level - core.base_level) / span
        )
