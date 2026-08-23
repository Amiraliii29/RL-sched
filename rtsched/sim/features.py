"""Observation encoding shared by both reinforcement-learning environments.

Cores are encoded as a ``(n_cores, CORE_FEATURES)`` matrix rather than a flat vector, and every
feature is a ratio.  A network that scores each row with shared weights and reduces over rows is
therefore permutation-invariant and core-count agnostic: one agent trained at 8 cores transfers
to 16 and 32 without retraining or reshaping.

Ratios that would otherwise sit near zero are amplified before clipping, never after, so every
value stays inside the observation space the environments declare.
"""

from __future__ import annotations

import numpy as np

from rtsched.policies.base import OnlineObservation

CORE_FEATURES = 8
MAPPING_TASK_FEATURES = 5
ONLINE_JOB_FEATURES = 6


def encode_online_cores(obs: OnlineObservation, max_level: int) -> np.ndarray:
    horizon = max(1, obs.horizon)
    work = max(1, obs.job_work_ticks)
    rows = [
        (
            core.utilization,
            core.base_level / max_level,
            core.level / max_level,
            min(1.0, core.slack_ticks * 10.0 / horizon),
            min(1.0, core.slack_ticks / work),
            min(1.0, core.soft_backlog_ticks / work),
            min(1.0, core.n_soft_pending / 8.0),
            float(core.busy),
        )
        for core in obs.cores
    ]
    return np.asarray(rows, dtype=np.float32)


def encode_online_job(obs: OnlineObservation) -> np.ndarray:
    horizon = max(1, obs.horizon)
    work = max(1, obs.job_work_ticks)
    return np.asarray(
        [
            min(1.0, work * 10.0 / horizon),
            np.clip(obs.laxity / work, -5.0, 5.0),
            min(1.0, (obs.job_deadline - obs.now) * 10.0 / horizon),
            obs.job_weight,
            obs.chip_power_w / max(1e-9, obs.tdp_w),
            obs.now / horizon,
        ],
        dtype=np.float32,
    )


def encode_mapping_cores(
    util_at_level: np.ndarray, level_idx: np.ndarray, levels: np.ndarray, n_placed: np.ndarray
) -> np.ndarray:
    """Core rows for the offline mapping environment, in the same layout as the online one."""
    max_level = levels[-1]
    n_cores = util_at_level.shape[0]
    chosen = util_at_level[np.arange(n_cores), level_idx]
    return np.column_stack(
        [
            np.minimum(1.0, chosen),
            levels[level_idx] / max_level,
            levels[level_idx] / max_level,
            np.maximum(0.0, 1.0 - chosen),
            np.maximum(0.0, 1.0 - util_at_level[:, -1]),
            np.zeros(n_cores),
            np.minimum(1.0, n_placed / 8.0),
            (util_at_level[:, -1] > 1.0).astype(np.float64),
        ]
    ).astype(np.float32)


def encode_mapping_task(
    utilization: float, remaining: int, total: int, boost_phase: bool, phase_progress: float
) -> np.ndarray:
    """Pending-task features plus which phase of the mapping episode we are in.

    The episode places every task first and only then sets each core's frequency boost, so the
    agent needs to know which decision it is being asked for.
    """
    return np.asarray(
        [
            utilization,
            remaining / max(1, total),
            1.0 - remaining / max(1, total),
            float(boost_phase),
            phase_progress,
        ],
        dtype=np.float32,
    )
