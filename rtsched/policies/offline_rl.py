"""Learned mapping policy -- the phase-2 replacement for the genetic algorithm.

Implements the same :class:`~rtsched.policies.base.OfflinePolicy` protocol, so the runner, the
metrics and every figure treat it exactly like the baseline.  Inference is one greedy rollout of
:class:`~rtsched.sim.env.MappingEnv` over the given task set: the agent places tasks in
decreasing-utilization order, choosing a core and a frequency boost per step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rtsched.config import ExperimentConfig
from rtsched.model.platform import Platform
from rtsched.model.taskset import TaskSet
from rtsched.policies.base import Allocation
from rtsched.rl import agents


class RLMappingPolicy:
    name = "rl"

    def __init__(self, checkpoint: str | Path, cfg: ExperimentConfig, device: str = "auto"):
        from rtsched.sim.env import MappingEnv

        self.policy, self.spec, self.meta, self.device = agents.load(Path(checkpoint), device)
        self.cfg = cfg
        self.actor = agents.GreedyActor(self.policy, self.spec, self.device)
        self._env = agents.wrap(MappingEnv(cfg, seeds=(cfg.seed,), reward_mode="terminal"), self.spec)

    def solve(self, taskset: TaskSet, platform: Platform, rng: np.random.Generator) -> Allocation:
        inner = self._env.env
        obs, mask = self._env.reset(options={"taskset": taskset})
        done = False
        while not done:
            obs, mask, _, done, _ = self._env.step(self.actor.act(obs, mask))

        allocation = inner.allocation()
        return Allocation(
            core_of_task=allocation.core_of_task,
            level_of_core=allocation.level_of_core,
            base_level_of_core=allocation.base_level_of_core,
            feasible=allocation.feasible,
            diagnostics={"solver": self.name, "run": self.meta.get("run", "rl-mapping")},
        )
