"""Gymnasium environments for the two decision points, built for phase 2.

Both wrap the *same* mechanism phase 1 uses -- :class:`~rtsched.analysis.objective.
AllocationObjective` for mapping and :class:`~rtsched.sim.engine.Engine` for dispatch -- so an
agent and a scripted policy are interchangeable and can be compared on identical episodes.
They are exercised in phase 1 by :mod:`rtsched.policies.scripted`, which replays the baseline
heuristics through the environment loop; that keeps the environments validated before any
learning code exists.

Actions are ``core * n_boost + boost`` (plus a trailing reject action online), and each
environment exposes ``action_masks()`` for masked policy-gradient methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - environments are optional in phase 1
    gym = None
    spaces = None

from rtsched.analysis.objective import AllocationObjective
from rtsched.config import ExperimentConfig
from rtsched.generation.taskgen import generate_taskset
from rtsched.model.platform import Platform
from rtsched.policies.base import Allocation, OnlineDecision
from rtsched.sim.engine import Engine
from rtsched.sim.features import (
    CORE_FEATURES,
    MAPPING_TASK_FEATURES,
    ONLINE_JOB_FEATURES,
    encode_mapping_cores,
    encode_mapping_task,
    encode_online_cores,
    encode_online_job,
)

_BASE = gym.Env if gym is not None else object


def _require_gym() -> None:
    if gym is None:
        raise ImportError("gymnasium is required for the RL environments: pip install gymnasium")


@dataclass
class OnlineReward:
    """Weights turning simulator events into the online agent's scalar reward."""

    qos: float = 1.0
    hard_miss: float = 10.0
    tdp: float = 2.0
    reject: float = 0.1


class MappingEnv(_BASE):
    """Offline phase, run as two consecutive decision phases over one episode.

    *Placement*: ``n_tasks`` steps assigning hard tasks to cores in decreasing-utilization order.
    *Boost*: ``n_cores`` further steps setting each core's frequency level above its schedulable
    floor.  Interleaving the two -- letting every placement also rewrite its core's boost -- means
    only the last task to land on a core determines its frequency, which leaves the DVFS decision
    with no usable credit assignment.

    Rewards are potential-based shaping of ``-AllocationObjective.cost``, the exact quantity the
    genetic-algorithm baseline minimises, so the two optimise the same objective.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: ExperimentConfig,
        seeds: tuple[int, ...] | None = None,
        reward_mode: str = "shaped",
    ):
        _require_gym()
        if reward_mode not in ("shaped", "terminal"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.cfg = cfg
        self.platform = Platform(cfg.platform)
        self.seeds = seeds or (cfg.seed,)
        self.reward_mode = reward_mode
        self.n_cores = cfg.platform.n_cores
        self.n_boost = cfg.ga.level_boost_max + 1

        self.action_space = spaces.Discrete(self.n_cores * self.n_boost)
        self.observation_space = spaces.Dict(
            {
                "cores": spaces.Box(-5.0, 5.0, (self.n_cores, CORE_FEATURES), np.float32),
                "task": spaces.Box(-5.0, 5.0, (MAPPING_TASK_FEATURES,), np.float32),
            }
        )
        self._episode = 0

    def reset(self, *, seed=None, options=None):
        """``options={"taskset": ts}`` scores a given task set instead of generating one.

        Inference goes through this same method, so the observation encoding, action layout and
        masking are identical to training -- no lookalike wrapper to drift out of sync.
        """
        super().reset(seed=seed)
        draw = self.seeds[self._episode % len(self.seeds)]
        self._episode += 1

        given = (options or {}).get("taskset")
        self.taskset = given if given is not None else generate_taskset(
            self.n_cores, self.cfg.taskgen, draw
        )
        self.objective = AllocationObjective(self.taskset, self.platform, self.cfg.ga)
        self.order = np.argsort(-self.objective.task_u)
        self.assign = np.full(self.objective.n_tasks, -1, dtype=np.int32)
        self.boost = np.zeros(self.n_cores, dtype=np.int32)
        self.pointer = 0
        self._potential = -self._partial_cost()
        return self._observation(), {}

    @property
    def n_steps(self) -> int:
        return self.objective.n_tasks + self.n_cores

    @property
    def in_boost_phase(self) -> bool:
        return self.pointer >= self.objective.n_tasks

    def step(self, action: int):
        core, boost = divmod(int(action), self.n_boost)
        if self.in_boost_phase:
            self.boost[self.pointer - self.objective.n_tasks] = boost
        else:
            self.assign[int(self.order[self.pointer])] = core
        self.pointer += 1

        done = self.pointer >= self.n_steps
        potential = -self._partial_cost()
        reward = potential - self._potential if self.reward_mode == "shaped" else 0.0
        self._potential = potential

        info: dict = {}
        if done:
            if self.reward_mode == "terminal":
                reward = -self.objective.cost(self.assign, self.boost)
            allocation = self.objective.to_allocation(self.assign, self.boost, {"solver": "rl"})
            info = {"allocation": allocation, "feasible": allocation.feasible, "cost": -potential}
        return self._observation(), float(reward), done, False, info

    def _partial_cost(self) -> float:
        placed = self.assign >= 0
        return self.objective.cost(self.assign, self.boost, placed if not placed.all() else None)

    def action_masks(self) -> np.ndarray:
        """Placement phase: cores that can host the task. Boost phase: the current core's levels."""
        mask = np.zeros(self.n_cores * self.n_boost, dtype=bool)
        if self.pointer >= self.n_steps:
            return np.ones_like(mask)

        if self.in_boost_phase:
            core = self.pointer - self.objective.n_tasks
            mask[core * self.n_boost : (core + 1) * self.n_boost] = True
            return mask

        placed = self.assign >= 0
        util = np.bincount(
            self.assign[placed], weights=self.objective.u_at_level[placed, -1], minlength=self.n_cores
        )
        index = int(self.order[self.pointer])
        fits = util + self.objective.u_at_level[index, -1] <= 1.0 + 1e-9
        if not fits.any():
            fits = np.ones(self.n_cores, dtype=bool)
        # Boost is decided in its own phase; force sub-action zero while placing.
        mask.reshape(self.n_cores, self.n_boost)[:, 0] = fits
        return mask

    def allocation(self) -> Allocation:
        return self.objective.to_allocation(self.assign, self.boost, {"solver": "rl"})

    def _observation(self):
        placed = self.assign >= 0
        util = np.column_stack(
            [
                np.bincount(
                    self.assign[placed], weights=self.objective.u_at_level[placed, l],
                    minlength=self.n_cores,
                )
                for l in range(self.objective.n_levels)
            ]
        )
        ok = util <= 1.0 + 1e-9
        level_idx = np.where(ok.any(axis=1), ok.argmax(axis=1), self.objective.n_levels - 1)
        level_idx = np.clip(level_idx + self.boost, 0, self.objective.n_levels - 1)
        n_placed = np.bincount(self.assign[placed], minlength=self.n_cores)

        n_tasks = self.objective.n_tasks
        remaining = max(0, n_tasks - self.pointer)
        task_u = float(self.objective.task_u[int(self.order[self.pointer])]) if remaining else 0.0
        boost_phase = self.in_boost_phase
        progress = (self.pointer - n_tasks) / self.n_cores if boost_phase else self.pointer / max(1, n_tasks)

        return {
            "cores": encode_mapping_cores(util, level_idx, self.objective.levels, n_placed),
            "task": encode_mapping_task(task_u, remaining, n_tasks, boost_phase, progress),
        }


class OnlineEnv(_BASE):
    """Online phase: dispatch each soft aperiodic arrival to a core at a chosen frequency.

    One episode is one hyperperiod.  Reward is credited between decisions: the QoS of soft jobs
    that completed, minus penalties for hard misses and time spent above TDP.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: ExperimentConfig,
        allocation_fn=None,
        seeds: tuple[int, ...] | None = None,
        reward: OnlineReward | None = None,
    ):
        _require_gym()
        self.cfg = cfg
        self.platform = Platform(cfg.platform)
        self.seeds = seeds or (cfg.seed,)
        self.reward_cfg = reward or OnlineReward()
        self.n_cores = cfg.platform.n_cores
        self.n_boost = len(self.platform.levels)
        self.allocation_fn = allocation_fn or self._default_allocation

        self.action_space = spaces.Discrete(self.n_cores * self.n_boost + 1)
        self.observation_space = spaces.Dict(
            {
                "cores": spaces.Box(-5.0, 5.0, (self.n_cores, CORE_FEATURES), np.float32),
                "job": spaces.Box(-5.0, 5.0, (ONLINE_JOB_FEATURES,), np.float32),
            }
        )
        self._episode = 0

    @property
    def reject_action(self) -> int:
        return self.n_cores * self.n_boost

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        draw = self.seeds[self._episode % len(self.seeds)]
        self._episode += 1

        self.taskset = generate_taskset(self.n_cores, self.cfg.taskgen, draw)
        self.alloc = self.allocation_fn(self.taskset, self.platform, draw)
        self.engine = Engine(self.taskset, self.platform, self.alloc, _NullPolicy(), self.cfg.sim)
        self.loop = self.engine.iter_decisions()
        self._seen_soft = 0
        self._seen_misses = 0
        self._seen_violation = 0
        self.result = None

        try:
            self.job, self.obs = next(self.loop)
        except StopIteration as stop:
            self.result = stop.value
            self.job = self.obs = None
        return self._observation(), {}

    def step(self, action: int):
        decision = self.decode(int(action))
        try:
            self.job, self.obs = self.loop.send(decision)
            done = False
        except StopIteration as stop:
            self.result = stop.value
            self.job = self.obs = None
            done = True

        reward = self._collect_reward(rejected=not decision.accept)
        info = {"result": self.result} if done else {}
        return self._observation(), reward, done, False, info

    def decode(self, action: int) -> OnlineDecision:
        if action == self.reject_action:
            return OnlineDecision.reject()
        core, level_idx = divmod(action, self.n_boost)
        return OnlineDecision(True, core, self.platform.levels[level_idx])

    def action_masks(self) -> np.ndarray:
        """Forbid levels below a core's floor; rejecting is always legal."""
        from rtsched.rl.encoding import online_action_mask

        if self.obs is None:
            mask = np.zeros(self.n_cores * self.n_boost + 1, dtype=bool)
            mask[self.reject_action] = True
            return mask
        return online_action_mask(self.obs, self.platform.levels)

    def _collect_reward(self, rejected: bool) -> float:
        w = self.reward_cfg
        engine = self.engine
        completed = engine.completed_soft[self._seen_soft :]
        self._seen_soft = len(engine.completed_soft)

        misses = engine.hard_misses - self._seen_misses
        self._seen_misses = engine.hard_misses
        violation = engine.tdp_violation_ticks - self._seen_violation
        self._seen_violation = engine.tdp_violation_ticks

        reward = w.qos * sum(job.qos() * job.weight for job in completed)
        reward -= w.hard_miss * misses
        reward -= w.tdp * violation / max(1, engine.horizon)
        if rejected:
            reward -= w.reject
        return float(reward)

    def _observation(self):
        if self.obs is None:
            return {
                "cores": np.zeros((self.n_cores, CORE_FEATURES), np.float32),
                "job": np.zeros(ONLINE_JOB_FEATURES, np.float32),
            }
        return {
            "cores": encode_online_cores(self.obs, self.platform.max_level),
            "job": encode_online_job(self.obs),
        }

    def _default_allocation(self, taskset, platform, seed) -> Allocation:
        from rtsched.policies.registry import make_offline

        rng = np.random.default_rng(seed + 10_000)
        return make_offline(self.cfg.offline_policy, self.cfg).solve(taskset, platform, rng)


class _NullPolicy:
    name = "external"

    def reset(self, taskset, platform, allocation) -> None:
        return None

    def on_arrival(self, job, obs) -> OnlineDecision:
        return OnlineDecision.reject()
