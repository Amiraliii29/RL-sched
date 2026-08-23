from __future__ import annotations

import numpy as np
import pytest

from rtsched.config import TaskGenConfig
from rtsched.generation.taskgen import generate_taskset
from rtsched.generation.uunifast import uunifast, uunifast_discard


@pytest.mark.parametrize("n,total", [(4, 2.0), (16, 8.0), (64, 24.0)])
def test_uunifast_sums_to_target(n, total):
    utils = uunifast(n, total, np.random.default_rng(0))
    assert len(utils) == n
    assert sum(utils) == pytest.approx(total)


def test_uunifast_discard_respects_cap():
    utils = uunifast_discard(20, 15.0, np.random.default_rng(1), u_max=1.0)
    assert max(utils) <= 1.0
    assert sum(utils) == pytest.approx(15.0, rel=1e-6)


def test_uunifast_discard_rejects_impossible_target():
    with pytest.raises(ValueError):
        uunifast_discard(4, 5.0, np.random.default_rng(0))


@pytest.mark.parametrize("cores", [8, 16, 32])
def test_taskset_matches_requested_utilization(cores):
    cfg = TaskGenConfig(utilization_per_core=0.75)
    ts = generate_taskset(cores, cfg, seed=3)
    assert len(ts.periodic) == cores * cfg.tasks_per_core
    assert ts.total_utilization == pytest.approx(cores * 0.75, rel=0.02)


def test_hyperperiod_is_bounded_and_divides_every_period():
    ts = generate_taskset(16, TaskGenConfig(), seed=0)
    assert ts.hyperperiod == 200_000
    assert all(ts.hyperperiod % t.period == 0 for t in ts.periodic)


def test_generation_is_deterministic():
    a = generate_taskset(8, TaskGenConfig(), seed=7)
    b = generate_taskset(8, TaskGenConfig(), seed=7)
    assert a.periodic == b.periodic
    assert a.aperiodic == b.aperiodic


def test_soft_jobs_can_finish_inside_the_hyperperiod():
    ts = generate_taskset(8, TaskGenConfig(), seed=0)
    assert ts.aperiodic
    assert all(a.deadline <= ts.hyperperiod for a in ts.aperiodic)
