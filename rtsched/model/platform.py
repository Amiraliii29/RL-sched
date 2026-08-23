"""Homogeneous multicore platform with per-core DVFS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from rtsched.config import PlatformConfig
from rtsched.power.model import PowerModel


@dataclass(frozen=True)
class Platform:
    cfg: PlatformConfig

    @property
    def n_cores(self) -> int:
        return self.cfg.n_cores

    @property
    def levels(self) -> tuple[int, ...]:
        return self.cfg.freq_levels

    @property
    def max_level(self) -> int:
        return self.levels[-1]

    @property
    def min_level(self) -> int:
        return self.levels[0]

    @cached_property
    def power(self) -> PowerModel:
        return PowerModel(self.cfg.power, self.levels, self.cfg.n_cores)

    def level_index(self, level: int) -> int:
        return self.levels.index(level)

    def level_at(self, index: int) -> int:
        return self.levels[min(max(index, 0), len(self.levels) - 1)]
