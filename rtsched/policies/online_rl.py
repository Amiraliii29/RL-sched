"""Learned dispatch policy -- the phase-2 replacement for the slack-stealing heuristic.

Implements :class:`~rtsched.policies.base.OnlinePolicy`, so the engine drives it exactly as it
drives the baseline: no environment in the loop at inference.  Observations and masks are built
with :mod:`rtsched.rl.encoding`, the same code the training wrapper uses.
"""

from __future__ import annotations

from pathlib import Path

from rtsched.config import ExperimentConfig
from rtsched.model.platform import Platform
from rtsched.model.taskset import TaskSet
from rtsched.policies.base import Allocation, OnlineDecision, OnlineObservation
from rtsched.rl import agents
from rtsched.rl.encoding import encode_online
from rtsched.model.task import Job


class RLOnlinePolicy:
    name = "rl"

    def __init__(
        self,
        checkpoint: str | Path,
        cfg: ExperimentConfig,
        device: str = "auto",
        deterministic: bool = False,
    ):
        self.policy, self.spec, self.meta, self.device = agents.load(Path(checkpoint), device)
        self.actor = agents.GreedyActor(
            self.policy, self.spec, self.device, deterministic=deterministic, seed=cfg.seed
        )
        self.seed = cfg.seed
        self.levels = cfg.platform.freq_levels
        self.max_level = self.levels[-1]
        self.reject_action = self.spec.max_cores * self.spec.n_sub

    def reset(self, taskset: TaskSet, platform: Platform, allocation: Allocation) -> None:
        self.levels = platform.levels
        self.max_level = platform.max_level
        self.actor.reseed(self.seed)

    def on_arrival(self, job: Job, obs: OnlineObservation) -> OnlineDecision:
        encoded, mask = encode_online(obs, self.max_level, self.spec.max_cores, self.levels)
        action = self.actor.act(encoded, mask)
        if action == self.reject_action:
            return OnlineDecision.reject()
        core, level_index = divmod(action, self.spec.n_sub)
        return OnlineDecision(True, core, self.levels[level_index])
