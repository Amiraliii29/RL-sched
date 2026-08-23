"""One experiment = generate -> solve offline -> simulate -> summarise.

This is the single code path every policy goes through, phase 1 and phase 2 alike.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from rtsched.config import ExperimentConfig, PlatformConfig, SimConfig, TaskGenConfig
from rtsched.generation.taskgen import generate_taskset
from rtsched.metrics.collect import summarize
from rtsched.model.platform import Platform
from rtsched.model.taskset import TaskSet
from rtsched.policies.base import Allocation
from rtsched.policies.registry import make_offline, make_online
from rtsched.sim.engine import simulate
from rtsched.sim.trace import SimResult


@dataclass
class RunOutput:
    config: ExperimentConfig
    taskset: TaskSet
    platform: Platform
    allocation: Allocation
    result: SimResult
    summary: dict


def build_config(
    n_cores: int,
    utilization_per_core: float,
    seed: int,
    offline: str = "ga",
    online: str = "slack",
    record_trace: bool = False,
    taskgen: TaskGenConfig | None = None,
    platform_cfg: PlatformConfig | None = None,
) -> ExperimentConfig:
    base_platform = platform_cfg or PlatformConfig()
    base_taskgen = taskgen or TaskGenConfig()
    return ExperimentConfig(
        platform=replace(base_platform, n_cores=n_cores),
        taskgen=replace(base_taskgen, utilization_per_core=utilization_per_core),
        sim=SimConfig(record_power_trace=record_trace),
        seed=seed,
        offline_policy=offline,
        online_policy=online,
    )


def run_experiment(cfg: ExperimentConfig) -> RunOutput:
    if cfg.offline_policy == "rl" or cfg.online_policy == "rl":
        from rtsched.rl.envs import _pin_torch_to_one_thread

        _pin_torch_to_one_thread()

    platform = Platform(cfg.platform)
    taskset = generate_taskset(cfg.platform.n_cores, cfg.taskgen, cfg.seed)

    rng = np.random.default_rng(cfg.seed + 10_000)
    offline = make_offline(cfg.offline_policy, cfg)
    started = time.perf_counter()
    allocation = offline.solve(taskset, platform, rng)
    solve_s = time.perf_counter() - started

    online = make_online(cfg.online_policy, cfg)
    started = time.perf_counter()
    result = simulate(taskset, platform, allocation, online, cfg.sim)
    sim_s = time.perf_counter() - started

    summary = {
        "offline_policy": cfg.offline_policy,
        "online_policy": cfg.online_policy,
        "policy": f"{cfg.offline_policy}+{cfg.online_policy}",
        "n_cores": cfg.platform.n_cores,
        "u_per_core": cfg.taskgen.utilization_per_core,
        "seed": cfg.seed,
        "n_tasks": len(taskset.periodic),
        "total_utilization": taskset.total_utilization,
        "solve_seconds": solve_s,
        "sim_seconds": sim_s,
        **summarize(result, allocation),
    }
    return RunOutput(cfg, taskset, platform, allocation, result, summary)
