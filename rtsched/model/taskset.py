"""Container for a generated workload."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

from rtsched.model.task import AperiodicTask, PeriodicTask


@dataclass(frozen=True)
class TaskSet:
    periodic: tuple[PeriodicTask, ...]
    aperiodic: tuple[AperiodicTask, ...]

    @property
    def hyperperiod(self) -> int:
        return lcm(*(t.period for t in self.periodic)) if self.periodic else 1

    @property
    def total_utilization(self) -> float:
        return sum(t.utilization for t in self.periodic)

    @property
    def aperiodic_utilization(self) -> float:
        h = self.hyperperiod
        return sum(t.exec_time for t in self.aperiodic) / h if h else 0.0

    @property
    def max_period(self) -> int:
        return max((t.period for t in self.periodic), default=1)

    def by_id(self) -> dict[int, PeriodicTask]:
        return {t.tid: t for t in self.periodic}
