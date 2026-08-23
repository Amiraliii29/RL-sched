from __future__ import annotations

import pytest

from rtsched.analysis.schedulability import (
    core_utilization,
    edf_feasible,
    exec_ticks,
    min_feasible_level,
)
from rtsched.analysis.slack import available_slack, ceil_div
from rtsched.model.task import Job, JobKind, PeriodicTask

LEVELS = (40, 50, 60, 70, 80, 90, 100)


def task(tid, period, wcet, deadline=None):
    return PeriodicTask(tid=tid, period=period, deadline=deadline or period, wcet=wcet)


def test_exec_time_inflates_at_lower_frequency():
    t = task(0, 1000, 300)
    assert exec_ticks(t, 100) == 300
    assert exec_ticks(t, 50) == 600
    assert exec_ticks(t, 60) == 500


def test_edf_feasible_at_the_utilization_bound():
    tasks = [task(0, 1000, 500), task(1, 1000, 500)]
    assert edf_feasible(tasks, 100)
    assert not edf_feasible(tasks, 90)


def test_min_feasible_level_is_the_lowest_that_works():
    tasks = [task(0, 1000, 400)]
    level = min_feasible_level(tasks, LEVELS)
    assert level == 40
    assert core_utilization(tasks, level) <= 1.0


def test_min_feasible_level_returns_none_when_overloaded():
    assert min_feasible_level([task(0, 100, 60), task(1, 100, 60)], LEVELS) is None


def test_constrained_deadlines_use_demand_analysis():
    tasks = [task(0, 1000, 400, deadline=500), task(1, 1000, 400, deadline=500)]
    assert core_utilization(tasks, 100) == pytest.approx(0.8)
    assert not edf_feasible(tasks, 100)


def test_slack_is_zero_on_a_fully_loaded_core():
    tasks = [task(0, 1000, 1000)]
    job = Job(0, 0, JobKind.HARD, 0, 1000, 100_000, 100_000)
    assert available_slack([job], tasks, {0: 1000}, 0, 10_000, 100) == 0


def test_slack_equals_idle_time_on_a_half_loaded_core():
    tasks = [task(0, 1000, 500)]
    job = Job(0, 0, JobKind.HARD, 0, 1000, 50_000, 50_000)
    assert available_slack([job], tasks, {0: 1000}, 0, 1000, 100) == 500


def test_slack_charges_per_job_rounding():
    """Two jobs of 3 ticks at level 40 cost ceil(300/40) each, not ceil(600/40)."""
    tasks: list = []
    jobs = [
        Job(0, 0, JobKind.HARD, 0, 100, 300, 300),
        Job(1, 1, JobKind.HARD, 0, 100, 300, 300),
    ]
    assert ceil_div(600, 40) == 15
    assert available_slack(jobs, tasks, {}, 0, 100, 40) == 100 - 2 * ceil_div(300, 40)


def test_slack_accounts_for_future_releases():
    tasks = [task(0, 1000, 900)]
    assert available_slack([], tasks, {0: 0}, 0, 1000, 100) == 100
