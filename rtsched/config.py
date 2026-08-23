"""Immutable configuration objects shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

TICKS_PER_MS = 1_000
"""Simulator time base: one tick is one microsecond."""

SECONDS_PER_TICK = 1e-6

LEVEL_DEN = 100
"""Frequency levels are integers out of ``LEVEL_DEN`` so all work arithmetic stays exact."""


@dataclass(frozen=True)
class PowerConfig:
    """Per-core power model parameters (see :mod:`rtsched.power.model`)."""

    static_w: float = 0.5
    dyn_max_w: float = 4.0
    uncore_w: float = 2.0
    v_min: float = 0.70
    v_max: float = 1.10
    tdp_scale: float = 0.8

@dataclass(frozen=True)
class PlatformConfig:
    n_cores: int = 8
    freq_levels: tuple[int, ...] = (40, 50, 60, 70, 80, 90, 100)
    power: PowerConfig = field(default_factory=PowerConfig)


@dataclass(frozen=True)
class TaskGenConfig:
    """Synthetic workload parameters."""

    utilization_per_core: float = 0.5
    tasks_per_core: int = 4
    periods_ms: tuple[int, ...] = (10, 20, 25, 40, 50, 100, 200)
    deadline_ratio: float = 1.0
    soft_load_ratio: float = 0.10
    soft_exec_ms: tuple[float, float] = (1.0, 5.0)
    soft_deadline_factor: tuple[float, float] = (3.0, 8.0)
    soft_weight: tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True)
class SimConfig:
    slack_horizon_factor: int = 0
    """Truncate the slack scan at this multiple of a core's longest period; 0 scans exactly."""

    record_power_trace: bool = False


@dataclass(frozen=True)
class GAConfig:
    population: int = 60
    generations: int = 80
    tournament: int = 3
    elitism: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float = 0.05
    local_search_rate: float = 0.25
    level_boost_max: int = 2
    w_infeasible: float = 1e3
    w_tdp: float = 50.0
    w_tdp_peak: float = 10.0
    w_energy: float = 1.0
    w_slack: float = 2.0
    w_balance: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    platform: PlatformConfig
    taskgen: TaskGenConfig
    sim: SimConfig = field(default_factory=SimConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    seed: int = 0
    offline_policy: str = "ga"
    online_policy: str = "slack"
