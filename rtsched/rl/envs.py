"""Picklable environment factories and the offline-allocation bank.

Workers are spawned, so every factory has to survive pickling -- hence module-level dataclasses
rather than closures.  The bank exists because ``OnlineEnv.reset`` would otherwise re-solve the
offline mapping on every episode, which costs more than the episode itself; allocations are
solved once per ``(cores, utilization, seed)`` and reused.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rtsched.experiments.pipeline import build_config
from rtsched.generation.taskgen import generate_taskset
from rtsched.model.platform import Platform
from rtsched.policies.base import Allocation
from rtsched.rl.agents import AgentSpec, wrap
from rtsched.sim.env import MappingEnv, OnlineEnv

TRAIN_SEEDS = tuple(range(1_000, 1_512))
"""Training instances, disjoint from the evaluation seeds so the comparison measures
generalisation to unseen task sets rather than memorisation."""

CORE_COUNTS = (8, 16, 32)
UTILIZATIONS = (0.25, 0.5, 0.75, 0.85, 0.95, 1.0)


def _pin_torch_to_one_thread() -> None:
    """Keep worker processes single-threaded.

    Torch defaults to one thread per core *per process*, so a pool of 12 workers asks for well
    over a hundred threads on a 20-thread machine and spends most of its time contending.  These
    forward passes are single small matrices; intra-op parallelism buys nothing anyway.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


@dataclass(frozen=True)
class BankLookup:
    table: dict[int, Allocation]

    def __call__(self, taskset, platform, seed) -> Allocation:
        return self.table[int(seed)]


@dataclass(frozen=True)
class MappingEnvFactory:
    cores: int
    utilization: float
    seeds: tuple[int, ...]
    spec: AgentSpec

    def __call__(self):
        cfg = build_config(self.cores, self.utilization, seed=self.seeds[0])
        return wrap(MappingEnv(cfg, seeds=self.seeds, reward_mode="shaped"), self.spec)


@dataclass(frozen=True)
class OnlineEnvFactory:
    cores: int
    utilization: float
    seeds: tuple[int, ...]
    spec: AgentSpec
    allocations: dict[int, Allocation] = field(default_factory=dict)

    def __call__(self):
        cfg = build_config(self.cores, self.utilization, seed=self.seeds[0])
        env = OnlineEnv(cfg, allocation_fn=BankLookup(self.allocations), seeds=self.seeds)
        return wrap(env, self.spec)


def env_grid(n_envs: int) -> list[tuple[int, float, tuple[int, ...]]]:
    """The ``(cores, utilization, seeds)`` each worker will be given.

    Spread by striding rather than ``i % len(grid)``, which silently truncates: with fewer
    workers than grid points it only ever reaches the first few, so a 12-worker run would never
    see a 32-core platform.

    Shared with the allocation bank so the bank solves exactly the instances that will be
    requested -- the full cross product is ``n_envs`` times more work, almost all unused.
    """
    grid = [(c, u) for c in CORE_COUNTS for u in UTILIZATIONS]
    assignments = []
    for i in range(n_envs):
        cores, u = grid[(i * len(grid)) // n_envs] if n_envs < len(grid) else grid[i % len(grid)]
        assignments.append((cores, u, TRAIN_SEEDS[i::n_envs]))
    return assignments


def bank_points(n_envs: int) -> list[tuple[int, float, int]]:
    return [(cores, u, seed) for cores, u, seeds in env_grid(n_envs) for seed in seeds]


def build_factories(kind: str, spec: AgentSpec, n_envs: int, bank: "AllocationBank | None" = None):
    """One factory per worker, over the assignment :func:`env_grid` defines."""
    factories = []
    for cores, u, seeds in env_grid(n_envs):
        if kind == "mapping":
            factories.append(MappingEnvFactory(cores, u, seeds, spec))
            continue

        # Restrict to seeds the bank actually holds, rather than failing mid-episode.
        available = tuple(s for s in seeds if (cores, u, s) in bank.table)
        if not available:
            raise ValueError(f"allocation bank has no entries for {cores} cores at u={u}")
        table = {s: bank[(cores, u, s)] for s in available}
        factories.append(OnlineEnvFactory(cores, u, available, spec, table))
    return factories


class AllocationBank:
    """``(cores, utilization, seed) -> Allocation``, solved once and cached on disk."""

    def __init__(self, table: dict | None = None):
        self.table: dict[tuple[int, float, int], Allocation] = table or {}

    def __getitem__(self, key) -> Allocation:
        return self.table[key]

    def __len__(self) -> int:
        return len(self.table)

    @classmethod
    def build(cls, offline_policy: str, points, workers: int = 8, progress=None) -> "AllocationBank":
        """Solve exactly the ``(cores, utilization, seed)`` triples given."""
        points = [(c, u, s, offline_policy) for c, u, s in points]
        table = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (key, allocation) in enumerate(pool.map(_solve, points, chunksize=8), 1):
                table[key] = allocation
                if progress:
                    progress(i, len(points))
        return cls(table)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self.table))
        return path

    @classmethod
    def load(cls, path: Path) -> "AllocationBank":
        return cls(pickle.loads(path.read_bytes()))


def _solve(point):
    cores, u, seed, policy_name = point
    _pin_torch_to_one_thread()
    cfg = build_config(cores, u, seed, offline=policy_name)
    taskset = generate_taskset(cores, cfg.taskgen, seed)
    from rtsched.policies.registry import make_offline

    allocation = make_offline(policy_name, cfg).solve(
        taskset, Platform(cfg.platform), np.random.default_rng(seed + 10_000)
    )
    return (cores, u, seed), allocation
