"""Behavioural guarantees of the simulator itself."""

from __future__ import annotations

import pytest

from rtsched.config import LEVEL_DEN
from rtsched.experiments.pipeline import build_config, run_experiment
from rtsched.model.platform import Platform
from rtsched.policies.base import OnlineDecision
from rtsched.sim.engine import simulate

CONFIGS = [(8, 0.25), (8, 0.5), (8, 0.75), (16, 0.5), (16, 0.75), (32, 0.5), (32, 0.75)]


class _RejectAll:
    name = "reject"

    def reset(self, taskset, platform, allocation):
        pass

    def on_arrival(self, job, obs):
        return OnlineDecision.reject()


@pytest.mark.parametrize("cores,u", CONFIGS)
@pytest.mark.parametrize("seed", [0, 1])
def test_feasible_allocations_never_miss_a_hard_deadline(cores, u, seed):
    """The central guarantee: slack stealing may delay hard jobs but never make them late."""
    out = run_experiment(build_config(cores, u, seed))
    if out.allocation.feasible:
        assert out.summary["hard_miss_ratio"] == 0.0


@pytest.mark.parametrize("cores,u", [(8, 0.5), (16, 0.75)])
def test_soft_traffic_cannot_change_the_hard_schedule_outcome(cores, u):
    cfg = build_config(cores, u, seed=0)
    out = run_experiment(cfg)
    quiet = simulate(out.taskset, out.platform, out.allocation, _RejectAll(), cfg.sim)
    assert all(not j.missed for j in quiet.hard_jobs) == (out.summary["hard_miss_ratio"] == 0.0)


def test_every_job_is_recorded_exactly_once():
    out = run_experiment(build_config(8, 0.5, seed=0))
    expected = sum(out.taskset.hyperperiod // t.period for t in out.taskset.periodic)
    assert len(out.result.hard_jobs) == expected
    assert len(out.result.soft_jobs) == len(out.taskset.aperiodic)
    assert len({j.jid for j in out.result.jobs}) == len(out.result.jobs)


def test_cores_never_run_below_their_allocated_floor():
    out = run_experiment(build_config(16, 0.5, seed=0, record_trace=True))
    floors = out.allocation.level_of_core
    for core, _, _, level, _ in out.result.trace.core_segments:
        assert level >= floors[core]


def test_energy_matches_the_recorded_power_trace():
    out = run_experiment(build_config(8, 0.5, seed=0, record_trace=True))
    integrated = sum((end - start) * 1e-6 * watts for start, end, watts in out.result.trace.chip_segments)
    assert integrated == pytest.approx(out.result.energy_j, rel=1e-9)


def test_power_trace_tiles_the_hyperperiod_without_gaps():
    out = run_experiment(build_config(8, 0.5, seed=0, record_trace=True))
    segments = out.result.trace.chip_segments
    assert segments[0][0] == 0
    assert segments[-1][1] == out.result.horizon
    assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))


def test_completed_jobs_did_exactly_their_work():
    out = run_experiment(build_config(8, 0.75, seed=0))
    for job in out.result.jobs:
        if job.finish >= 0:
            assert job.start >= job.release
            assert job.finish > job.start or job.work_ticks == 0


def test_simulation_is_deterministic():
    timing = {"solve_seconds", "sim_seconds"}
    a = run_experiment(build_config(16, 0.75, seed=2)).summary
    b = run_experiment(build_config(16, 0.75, seed=2)).summary
    assert {k: v for k, v in a.items() if k not in timing} == {
        k: v for k, v in b.items() if k not in timing
    }


def test_tdp_clamp_keeps_boosts_inside_the_power_budget():
    out = run_experiment(build_config(8, 0.25, seed=0, record_trace=True))
    platform = Platform(out.config.platform)
    for _, _, watts in out.result.trace.chip_segments:
        assert watts <= platform.power.max_chip_power + 1e-9


def test_work_units_scale_with_the_frequency_denominator():
    out = run_experiment(build_config(8, 0.5, seed=0))
    task = out.taskset.periodic[0]
    assert task.work_units == task.wcet * LEVEL_DEN
