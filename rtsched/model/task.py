"""Task and job types.

Execution demand is carried as integer *work units* (``ticks * LEVEL_DEN``) so a core
running at frequency level ``L`` consumes exactly ``L`` work units per tick.  This keeps
every time computation in exact integer arithmetic across DVFS changes and preemptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rtsched.config import LEVEL_DEN


class JobKind(Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class PeriodicTask:
    """Hard periodic task with WCET expressed at the maximum frequency level."""

    tid: int
    period: int
    deadline: int
    wcet: int

    @property
    def utilization(self) -> float:
        return self.wcet / self.period

    @property
    def work_units(self) -> int:
        return self.wcet * LEVEL_DEN


@dataclass(frozen=True, slots=True)
class AperiodicTask:
    """Soft aperiodic arrival; ``deadline`` is absolute."""

    aid: int
    arrival: int
    exec_time: int
    deadline: int
    weight: float = 1.0

    @property
    def work_units(self) -> int:
        return self.exec_time * LEVEL_DEN

    @property
    def relative_deadline(self) -> int:
        return self.deadline - self.arrival


@dataclass(slots=True)
class Job:
    """A single released instance of a periodic or aperiodic task."""

    jid: int
    source_id: int
    kind: JobKind
    release: int
    abs_deadline: int
    work: int
    remaining: int
    core: int = -1
    start: int = -1
    finish: int = -1
    weight: float = 1.0
    accepted: bool = True

    @property
    def completed(self) -> bool:
        return self.finish >= 0

    @property
    def missed(self) -> bool:
        return not self.completed or self.finish > self.abs_deadline

    def qos(self) -> float:
        """Graded service quality in ``[0, 1]``: full credit on time, linear decay when late."""
        if not self.accepted or not self.completed:
            return 0.0
        if self.finish <= self.abs_deadline:
            return 1.0
        span = max(1, self.abs_deadline - self.release)
        return max(0.0, 1.0 - (self.finish - self.abs_deadline) / span)


@dataclass(slots=True)
class JobRecord:
    """Flat row emitted to the trace for every job the simulator saw."""

    jid: int
    source_id: int
    kind: str
    core: int
    release: int
    abs_deadline: int
    work_ticks: int
    start: int
    finish: int
    accepted: bool
    missed: bool
    qos: float
    weight: float

    @classmethod
    def of(cls, job: Job) -> "JobRecord":
        return cls(
            jid=job.jid,
            source_id=job.source_id,
            kind=job.kind.value,
            core=job.core,
            release=job.release,
            abs_deadline=job.abs_deadline,
            work_ticks=job.work // LEVEL_DEN,
            start=job.start,
            finish=job.finish,
            accepted=job.accepted,
            missed=job.missed,
            qos=job.qos(),
            weight=job.weight,
        )
