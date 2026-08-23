"""The policy seam.

Everything above this line is mechanism (EDF dispatch, slack stealing, DVFS actuation, power
accounting) and lives in the simulator.  Everything below is policy: *which* core and *what*
frequency.  Phase 1 ships genetic-algorithm and heuristic implementations; phase 2 adds
reinforcement-learning ones without touching the engine, the generator, or the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from rtsched.model.platform import Platform
from rtsched.model.task import Job
from rtsched.model.taskset import TaskSet


@dataclass(frozen=True)
class Allocation:
    """Result of the offline phase: a task-to-core map and a per-core frequency level."""

    core_of_task: dict[int, int]
    level_of_core: tuple[int, ...]
    base_level_of_core: tuple[int, ...]
    feasible: bool
    diagnostics: dict = field(default_factory=dict)

    def tasks_on(self, core: int) -> list[int]:
        return [tid for tid, c in self.core_of_task.items() if c == core]


@dataclass(frozen=True)
class CoreObservation:
    index: int
    level: int
    base_level: int
    max_level: int
    utilization: float
    slack_ticks: int
    soft_backlog_ticks: int
    n_soft_pending: int
    busy: bool


@dataclass(frozen=True)
class OnlineObservation:
    """State handed to the online policy at a soft-task arrival."""

    now: int
    horizon: int
    cores: tuple[CoreObservation, ...]
    chip_power_w: float
    tdp_w: float
    job_work_ticks: int
    job_deadline: int
    job_weight: float

    @property
    def laxity(self) -> int:
        return self.job_deadline - self.now - self.job_work_ticks


@dataclass(frozen=True)
class OnlineDecision:
    """``core < 0`` or ``accept=False`` rejects the job; ``level < 0`` keeps the core's level."""

    accept: bool
    core: int = -1
    level: int = -1

    @staticmethod
    def reject() -> "OnlineDecision":
        return OnlineDecision(accept=False)


@runtime_checkable
class OfflinePolicy(Protocol):
    name: str

    def solve(self, taskset: TaskSet, platform: Platform, rng: np.random.Generator) -> Allocation: ...


@runtime_checkable
class OnlinePolicy(Protocol):
    name: str

    def reset(self, taskset: TaskSet, platform: Platform, allocation: Allocation) -> None: ...

    def on_arrival(self, job: Job, obs: OnlineObservation) -> OnlineDecision: ...
