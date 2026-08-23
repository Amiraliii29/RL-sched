"""Replay phase-1 policies through the phase-2 environments.

This is what keeps :mod:`rtsched.sim.env` honest before any learning code exists: driving
``OnlineEnv`` with the slack-stealing heuristic must reproduce, job for job, what
``simulate()`` produces with the same heuristic.  The tests assert exactly that.
"""

from __future__ import annotations

import numpy as np

from rtsched.policies.base import Allocation, OnlinePolicy
from rtsched.sim.trace import SimResult


def greedy_mapping_action(env) -> int:
    """Worst-fit placement, then the lowest boost -- a ``MappingEnv`` action in either phase."""
    legal = np.flatnonzero(env.action_masks())
    if env.in_boost_phase:
        return int(legal[0])
    cores = legal // env.n_boost
    spare = env._observation()["cores"][cores, 4]
    return int(legal[int(np.argmax(spare))])


def run_mapping_env(env, action_fn=greedy_mapping_action) -> tuple[Allocation, float]:
    env.reset()
    total = 0.0
    done = False
    info: dict = {}
    while not done:
        _, reward, done, _, info = env.step(action_fn(env))
        total += reward
    return info["allocation"], total


def policy_action(env, policy: OnlinePolicy) -> int:
    """Encode the heuristic's decision in the environment's discrete action space."""
    decision = policy.on_arrival(env.job, env.obs)
    if not decision.accept:
        return env.reject_action
    level = decision.level if decision.level >= 0 else env.obs.cores[decision.core].level
    return decision.core * env.n_boost + env.platform.level_index(level)


def run_online_env(env, policy: OnlinePolicy) -> SimResult:
    env.reset()
    policy.reset(env.taskset, env.platform, env.alloc)
    done = env.job is None
    while not done:
        _, _, done, _, _ = env.step(policy_action(env, policy))
    return env.result
