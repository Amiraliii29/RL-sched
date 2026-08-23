"""Single source of truth for how observations and action masks are padded.

Training pads through :class:`~rtsched.rl.vecenv.PaddedEnv`; inference for the online agent
happens inside the engine's policy callback, with no environment in the loop.  Both call the
functions here, so the two paths cannot drift -- the classic way a reinforcement-learning system
scores well in training and badly in production.
"""

from __future__ import annotations

import numpy as np

from rtsched.policies.base import OnlineObservation
from rtsched.sim.features import encode_online_cores, encode_online_job


def pad_cores(cores: np.ndarray, max_cores: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the zero-padded core matrix and a boolean marking the real rows."""
    if cores.shape[0] > max_cores:
        raise ValueError(f"{cores.shape[0]} cores exceeds max_cores={max_cores}")
    padded = np.zeros((max_cores, cores.shape[1]), dtype=np.float32)
    padded[: cores.shape[0]] = cores
    valid = np.zeros(max_cores, dtype=bool)
    valid[: cores.shape[0]] = True
    return padded, valid


def pad_mask(
    inner: np.ndarray, n_cores: int, n_sub: int, max_cores: int, has_reject: bool
) -> np.ndarray:
    """Lift an environment-sized action mask into the padded action space."""
    mask = np.zeros(max_cores * n_sub + (1 if has_reject else 0), dtype=bool)
    limit = n_cores * n_sub
    mask[:limit] = inner[:limit]
    if has_reject:
        mask[-1] = inner[limit]
    return mask


def online_action_mask(obs: OnlineObservation, levels: tuple[int, ...]) -> np.ndarray:
    """Legal dispatches: any level at or above a core's floor, plus rejecting the job."""
    n_cores, n_sub = len(obs.cores), len(levels)
    mask = np.zeros(n_cores * n_sub + 1, dtype=bool)
    mask[-1] = True
    for core in obs.cores:
        for i, level in enumerate(levels):
            if level >= core.base_level:
                mask[core.index * n_sub + i] = True
    return mask


def encode_online(obs: OnlineObservation, max_level: int, max_cores: int, levels) -> tuple[dict, np.ndarray]:
    """Full padded observation and mask for a soft arrival, as the agent sees it."""
    cores, valid = pad_cores(encode_online_cores(obs, max_level), max_cores)
    mask = pad_mask(online_action_mask(obs, levels), len(obs.cores), len(levels), max_cores, True)
    return {"cores": cores, "context": encode_online_job(obs), "valid": valid}, mask
