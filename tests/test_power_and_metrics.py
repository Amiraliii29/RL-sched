from __future__ import annotations

import pytest

from rtsched.config import PlatformConfig
from rtsched.experiments.pipeline import build_config, run_experiment
from rtsched.metrics.collect import core_frame, jobs_frame, summarize, taskset_frame
from rtsched.model.platform import Platform
from rtsched.model.task import Job, JobKind

LEVELS = PlatformConfig().freq_levels


@pytest.fixture(scope="module")
def platform():
    return Platform(PlatformConfig(n_cores=8))


def test_power_is_monotone_in_frequency(platform):
    powers = [platform.power.core_power(l, True) for l in LEVELS]
    assert powers == sorted(powers)


def test_idle_power_is_static_only(platform):
    for level in LEVELS:
        assert platform.power.core_power(level, False) == platform.cfg.power.static_w


def test_tdp_sits_below_the_all_cores_maximum(platform):
    assert 0 < platform.power.tdp < platform.power.max_chip_power


def test_chip_power_adds_uncore_once(platform):
    levels = [platform.max_level] * platform.n_cores
    expected = platform.cfg.power.uncore_w + platform.n_cores * platform.power.core_power(
        platform.max_level, True
    )
    assert platform.power.chip_power(levels, [True] * platform.n_cores) == pytest.approx(expected)


def test_qos_is_one_on_time_and_decays_when_late():
    on_time = Job(0, 0, JobKind.SOFT, 0, 100, 100, 0, finish=90)
    exact = Job(1, 1, JobKind.SOFT, 0, 100, 100, 0, finish=100)
    half_late = Job(2, 2, JobKind.SOFT, 0, 100, 100, 0, finish=150)
    hopeless = Job(3, 3, JobKind.SOFT, 0, 100, 100, 0, finish=500)

    assert on_time.qos() == 1.0
    assert exact.qos() == 1.0
    assert half_late.qos() == pytest.approx(0.5)
    assert hopeless.qos() == 0.0


def test_rejected_and_unfinished_jobs_score_zero():
    rejected = Job(0, 0, JobKind.SOFT, 0, 100, 100, 0, accepted=False)
    unfinished = Job(1, 1, JobKind.SOFT, 0, 100, 100, 50)
    assert rejected.qos() == 0.0 and rejected.missed
    assert unfinished.qos() == 0.0 and unfinished.missed


def test_summary_ratios_stay_in_range():
    out = run_experiment(build_config(8, 0.75, seed=0))
    summary = summarize(out.result, out.allocation)
    for key in ("hard_miss_ratio", "soft_miss_ratio", "qos_mean", "acceptance_ratio",
                "tdp_violation_ratio"):
        assert 0.0 <= summary[key] <= 1.0


def test_frames_cover_the_whole_run():
    out = run_experiment(build_config(8, 0.5, seed=0))
    jobs = jobs_frame(out.result)
    cores = core_frame(out.result)
    tasks = taskset_frame(out.taskset, out.allocation)

    assert len(jobs) == len(out.result.jobs)
    assert len(cores) == out.config.platform.n_cores
    assert len(tasks) == len(out.taskset.periodic) + len(out.taskset.aperiodic)
    assert set(tasks["kind"]) == {"hard", "soft"}


def test_mean_core_power_reconstructs_chip_power():
    out = run_experiment(build_config(8, 0.5, seed=0))
    cores = core_frame(out.result)
    uncore = out.config.platform.power.uncore_w
    assert cores["mean_power_w"].sum() + uncore == pytest.approx(
        out.result.mean_chip_power_w, rel=1e-9
    )
