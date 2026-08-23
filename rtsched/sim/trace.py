"""Simulation output.

The engine emits only two things: a job record per released job and a piecewise-constant
power trace.  Every metric and every figure is derived from these, so adding a new metric
never requires re-running the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rtsched.model.task import JobRecord


@dataclass
class PowerTrace:
    """Piecewise-constant power segments, merged across steps with identical power."""

    core_segments: list[tuple[int, int, int, int, float]] = field(default_factory=list)
    chip_segments: list[tuple[int, int, float]] = field(default_factory=list)

    def add_core(self, core: int, start: int, end: int, level: int, watts: float) -> None:
        if self.core_segments:
            c, s, e, lv, w = self.core_segments[-1]
            if c == core and e == start and lv == level and w == watts:
                self.core_segments[-1] = (c, s, end, lv, w)
                return
        self.core_segments.append((core, start, end, level, watts))

    def add_chip(self, start: int, end: int, watts: float) -> None:
        if self.chip_segments and self.chip_segments[-1][1] == start and self.chip_segments[-1][2] == watts:
            s, _, w = self.chip_segments[-1]
            self.chip_segments[-1] = (s, end, w)
            return
        self.chip_segments.append((start, end, watts))


@dataclass
class SimResult:
    horizon: int
    jobs: list[JobRecord]
    energy_j: float
    peak_chip_power_w: float
    mean_chip_power_w: float
    tdp_w: float
    tdp_violation_ratio: float
    core_energy_j: list[float]
    core_busy_ratio: list[float]
    core_levels: list[int]
    steps: int
    trace: PowerTrace | None = None

    @property
    def hard_jobs(self) -> list[JobRecord]:
        return [j for j in self.jobs if j.kind == "hard"]

    @property
    def soft_jobs(self) -> list[JobRecord]:
        return [j for j in self.jobs if j.kind == "soft"]
