"""Phase-2 readiness: the environments must be interchangeable with the phase-1 code path.

If these pass, a learning agent can be dropped in behind the same interfaces and its results
are directly comparable to the genetic-algorithm baseline on identical episodes.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rtsched.experiments.pipeline import build_config, run_experiment
from rtsched.metrics.collect import summarize
from rtsched.policies.online_slack import SlackStealingPolicy
from rtsched.policies.scripted import greedy_mapping_action, run_mapping_env, run_online_env
from rtsched.sim.env import MappingEnv, OnlineEnv


def test_online_env_reproduces_the_scripted_simulation_exactly():
    cfg = build_config(16, 0.75, seed=0)
    reference = run_experiment(cfg)

    env = OnlineEnv(cfg, seeds=(cfg.seed,))
    replay = run_online_env(env, SlackStealingPolicy())

    assert summarize(replay, env.alloc) == summarize(reference.result, reference.allocation)
    assert [asdict(j) for j in replay.jobs] == [asdict(j) for j in reference.result.jobs]


@pytest.mark.parametrize("cores", [8, 16])
def test_online_env_obeys_the_gymnasium_contract(cores):
    env = OnlineEnv(build_config(cores, 0.5, seed=0))
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    done = env.job is None
    steps = 0
    while not done:
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        obs, reward, done, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward) and not truncated
        steps += 1
    assert steps == len(env.taskset.aperiodic)
    assert info["result"] is not None


def test_online_action_mask_never_allows_a_level_below_the_core_floor():
    env = OnlineEnv(build_config(8, 0.75, seed=0))
    env.reset()
    mask = env.action_masks()
    for action in np.flatnonzero(mask):
        if action == env.reject_action:
            continue
        decision = env.decode(int(action))
        assert decision.level >= env.obs.cores[decision.core].base_level


def test_online_rejecting_everything_yields_zero_qos():
    env = OnlineEnv(build_config(8, 0.5, seed=0))
    env.reset()
    done = env.job is None
    while not done:
        _, _, done, _, _ = env.step(env.reject_action)
    assert summarize(env.result, env.alloc)["qos_mean"] == 0.0


def test_mapping_env_terminal_reward_is_the_ga_objective():
    cfg = build_config(8, 0.5, seed=0)
    env = MappingEnv(cfg, reward_mode="terminal")
    allocation, reward = run_mapping_env(env)
    assert reward == pytest.approx(-env.objective.cost(env.assign, env.boost))
    assert set(allocation.core_of_task) == {t.tid for t in env.taskset.periodic}


def test_mapping_env_obeys_the_gymnasium_contract():
    env = MappingEnv(build_config(8, 0.5, seed=0))
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    done = False
    steps = 0
    while not done:
        assert env.action_masks().any()
        obs, reward, done, truncated, info = env.step(greedy_mapping_action(env))
        assert env.observation_space.contains(obs) and not truncated
        steps += 1
    assert steps == len(env.taskset.periodic) + env.n_cores
    assert info["allocation"].feasible


def test_mapping_env_mask_blocks_cores_that_cannot_fit_the_task():
    env = MappingEnv(build_config(8, 1.0, seed=0))
    env.reset()
    for _ in range(env.n_steps):
        mask = env.action_masks()
        assert mask.any()
        env.step(int(np.flatnonzero(mask)[0]))


def test_environments_are_seed_reproducible():
    cfg = build_config(8, 0.5, seed=3)
    first = run_online_env(OnlineEnv(cfg), SlackStealingPolicy())
    second = run_online_env(OnlineEnv(cfg), SlackStealingPolicy())
    assert [asdict(j) for j in first.jobs] == [asdict(j) for j in second.jobs]
