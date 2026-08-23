"""Per-core DVFS power model and the chip-level TDP budget.

A core at frequency level ``L`` (out of :data:`~rtsched.config.LEVEL_DEN`) runs at speed
``L / LEVEL_DEN`` and at supply voltage interpolated linearly between ``v_min`` and
``v_max``.  Dynamic power follows the usual ``C_eff * V^2 * f`` relation and is gated off
while the core is idle; static power is always paid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from rtsched.config import LEVEL_DEN, PowerConfig


@dataclass(frozen=True)
class PowerModel:
    cfg: PowerConfig
    levels: tuple[int, ...]
    n_cores: int

    @cached_property
    def _c_eff(self) -> float:
        return self.cfg.dyn_max_w / (self.cfg.v_max**2 * 1.0)

    def voltage(self, level: int) -> float:
        lo, hi = self.levels[0], self.levels[-1]
        if hi == lo:
            return self.cfg.v_max
        frac = (level - lo) / (hi - lo)
        return self.cfg.v_min + (self.cfg.v_max - self.cfg.v_min) * frac

    def core_power(self, level: int, active: bool) -> float:
        if not active:
            return self.cfg.static_w
        v = self.voltage(level)
        return self.cfg.static_w + self._c_eff * v * v * (level / LEVEL_DEN)

    def chip_power(self, levels, active) -> float:
        return self.cfg.uncore_w + sum(
            self.core_power(lv, act) for lv, act in zip(levels, active)
        )

    @cached_property
    def max_chip_power(self) -> float:
        top = self.levels[-1]
        return self.cfg.uncore_w + self.n_cores * self.core_power(top, True)

    @cached_property
    def tdp(self) -> float:
        return self.cfg.tdp_scale * self.max_chip_power

    def expected_chip_power(self, levels, busy_fractions) -> float:
        """Time-averaged chip power given each core's fraction of time executing."""
        total = self.cfg.uncore_w
        for lv, busy in zip(levels, busy_fractions):
            b = min(1.0, max(0.0, busy))
            total += b * self.core_power(lv, True) + (1.0 - b) * self.core_power(lv, False)
        return total
