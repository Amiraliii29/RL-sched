from __future__ import annotations

import numpy as np
import pytest

from rtsched.analysis.objective import AllocationObjective
from rtsched.analysis.schedulability import edf_feasible
from rtsched.config import ExperimentConfig, PlatformConfig, TaskGenConfig
from rtsched.experiments.pipeline import build_config, run_experiment
from rtsched.generation.taskgen import generate_taskset
from rtsched.model.platform import Platform
from rtsched.policies.registry import OFFLINE, ONLINE, make_offline, make_online


def _setup(cores=8, u=0.5, seed=0):
    cfg = build_config(cores, u, seed)
    return cfg, generate_taskset(cores, cfg.taskgen, seed), Platform(cfg.platform)


def _policy(factory, name, cfg):
    """Skip policies whose trained checkpoint has not been produced yet."""
    try:
        return factory(name, cfg)
    except FileNotFoundError as missing:
        pytest.skip(str(missing))


@pytest.mark.parametrize("name", sorted(OFFLINE))
def test_offline_policies_produce_consistent_allocations(name):
    cfg, taskset, platform = _setup()
    alloc = _policy(make_offline, name, cfg).solve(taskset, platform, np.random.default_rng(0))

    assert set(alloc.core_of_task) == {t.tid for t in taskset.periodic}
    assert all(0 <= c < platform.n_cores for c in alloc.core_of_task.values())
    assert len(alloc.level_of_core) == platform.n_cores
    for core in range(platform.n_cores):
        assert alloc.level_of_core[core] >= alloc.base_level_of_core[core]


@pytest.mark.parametrize("name", sorted(OFFLINE))
def test_offline_feasibility_flag_agrees_with_the_edf_test(name):
    cfg, taskset, platform = _setup(u=0.5)
    alloc = _policy(make_offline, name, cfg).solve(taskset, platform, np.random.default_rng(0))
    by_id = taskset.by_id()
    per_core_ok = all(
        edf_feasible([by_id[t] for t in alloc.tasks_on(c)], alloc.base_level_of_core[c])
        for c in range(platform.n_cores)
    )
    assert alloc.feasible == per_core_ok


def test_offline_policies_are_reproducible():
    cfg, taskset, platform = _setup()
    a = make_offline("ga", cfg).solve(taskset, platform, np.random.default_rng(5))
    b = make_offline("ga", cfg).solve(taskset, platform, np.random.default_rng(5))
    assert a.core_of_task == b.core_of_task
    assert a.level_of_core == b.level_of_core


def test_ga_beats_its_own_greedy_seed():
    cfg, taskset, platform = _setup(cores=16, u=0.5)
    objective = AllocationObjective(taskset, platform, cfg.ga)
    seed_cost = objective.cost(objective.greedy_assignment(), np.zeros(platform.n_cores, dtype=int))
    alloc = make_offline("ga", cfg).solve(taskset, platform, np.random.default_rng(0))
    assert alloc.diagnostics["fitness"] <= seed_cost


def test_ga_convergence_is_monotone():
    cfg, taskset, platform = _setup(cores=16)
    history = make_offline("ga", cfg).solve(
        taskset, platform, np.random.default_rng(0)
    ).diagnostics["history"]
    assert all(b <= a + 1e-12 for a, b in zip(history, history[1:]))


@pytest.mark.parametrize("name", sorted(ONLINE))
def test_online_policies_return_legal_decisions(name):
    cfg = build_config(8, 0.5, 0, online=name)
    _policy(make_online, name, cfg)
    out = run_experiment(cfg)
    for job in out.result.soft_jobs:
        assert job.accepted == (job.core >= 0)
        if job.accepted:
            assert 0 <= job.core < cfg.platform.n_cores


def test_disabling_dvfs_never_spends_more_energy():
    with_boost = run_experiment(build_config(16, 0.5, 0, online="slack")).summary
    without = run_experiment(build_config(16, 0.5, 0, online="slack_no_dvfs")).summary
    assert without["energy_j"] <= with_boost["energy_j"] + 1e-9


def test_registry_rejects_unknown_names():
    cfg = build_config(8, 0.5, 0)
    with pytest.raises(KeyError):
        make_offline("does-not-exist", cfg)
    with pytest.raises(KeyError):
        make_online("does-not-exist", cfg)


def test_objective_penalises_overload_and_rewards_slack():
    cfg = ExperimentConfig(platform=PlatformConfig(n_cores=4), taskgen=TaskGenConfig())
    taskset = generate_taskset(4, cfg.taskgen, seed=0)
    objective = AllocationObjective(taskset, Platform(cfg.platform), cfg.ga)
    balanced = objective.greedy_assignment()
    piled = np.zeros_like(balanced)
    zero = np.zeros(4, dtype=int)
    assert objective.cost(piled, zero) > objective.cost(balanced, zero)
