"""UUniFast utilization generation (Bini & Buttazzo, 2005)."""

from __future__ import annotations

import numpy as np


def uunifast(n: int, total_u: float, rng: np.random.Generator) -> list[float]:
    """Draw ``n`` utilizations summing to ``total_u``, uniformly over the valid simplex."""
    utils: list[float] = []
    remaining = total_u
    for i in range(1, n):
        nxt = remaining * rng.random() ** (1.0 / (n - i))
        utils.append(remaining - nxt)
        remaining = nxt
    utils.append(remaining)
    return utils


def uunifast_discard(
    n: int,
    total_u: float,
    rng: np.random.Generator,
    u_max: float = 1.0,
    max_tries: int = 1_000,
) -> list[float]:
    """UUniFast restricted to per-task utilization ``<= u_max``.

    Rejection sampling gets increasingly unlikely to succeed as ``total_u / n`` approaches
    ``u_max``, so a draw that never passes falls back to redistribution, which caps the
    offenders and spreads their excess over the remaining tasks without disturbing the total.
    """
    if total_u > n * u_max:
        raise ValueError(f"total utilization {total_u} exceeds {n} tasks x {u_max}")
    for _ in range(max_tries):
        utils = uunifast(n, total_u, rng)
        if all(u <= u_max for u in utils):
            return utils
    return _redistribute(uunifast(n, total_u, rng), u_max)


def _redistribute(utils: list[float], u_max: float, max_rounds: int = 100) -> list[float]:
    """Cap at ``u_max`` and push the overflow onto tasks that still have room."""
    u = np.asarray(utils, dtype=float)
    for _ in range(max_rounds):
        excess = float(np.clip(u - u_max, 0.0, None).sum())
        if excess <= 1e-12:
            break
        u = np.minimum(u, u_max)
        room = u_max - u
        capacity = room.sum()
        if capacity <= 1e-12:
            break
        u += room * (min(excess, capacity) / capacity)
    return list(np.minimum(u, u_max))
