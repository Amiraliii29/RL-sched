"""Genetic-algorithm mapping and DVFS assignment -- the phase-1 baseline solver.

A genome is a task-to-core assignment plus a per-core *boost*: how many frequency steps above
the core's lowest schedulable level to run.  Boosting costs power but buys spare capacity,
which the slack-stealing server later converts into soft-task QoS.  The trade-off itself lives
in :class:`~rtsched.analysis.objective.AllocationObjective`; this module only searches.
"""

from __future__ import annotations

import numpy as np

from rtsched.analysis.objective import AllocationObjective
from rtsched.config import GAConfig
from rtsched.model.platform import Platform
from rtsched.model.taskset import TaskSet
from rtsched.policies.base import Allocation


class GeneticMappingPolicy:
    name = "ga"

    def __init__(self, cfg: GAConfig | None = None):
        self.cfg = cfg or GAConfig()

    def solve(self, taskset: TaskSet, platform: Platform, rng: np.random.Generator) -> Allocation:
        cfg = self.cfg
        objective = AllocationObjective(taskset, platform, cfg)
        n, m = objective.n_tasks, platform.n_cores

        assign = np.empty((cfg.population, n), dtype=np.int32)
        assign[0] = objective.greedy_assignment()
        assign[1:] = rng.integers(0, m, size=(cfg.population - 1, n), dtype=np.int32)
        boost = rng.integers(0, cfg.level_boost_max + 1, size=(cfg.population, m), dtype=np.int32)
        boost[0] = 0

        fitness = self._evaluate(objective, assign, boost)
        history = [float(fitness.min())]

        for _ in range(cfg.generations):
            assign, boost = self._next_generation(objective, assign, boost, fitness, rng)
            self._local_search(objective, assign, boost, rng)
            fitness = self._evaluate(objective, assign, boost)
            history.append(float(fitness.min()))

        best = int(np.argmin(fitness))
        return objective.to_allocation(
            assign[best],
            boost[best],
            {"solver": self.name, "fitness": history[-1], "history": history},
        )

    @staticmethod
    def _evaluate(objective, assign, boost) -> np.ndarray:
        return np.array([objective.cost(a, b) for a, b in zip(assign, boost)])

    def _local_search(self, objective, assign, boost, rng) -> None:
        """Memetic step: crossover alone stalls once the task count grows past a few dozen.

        Each selected child tries one load-shedding move (shift a task off the busiest core)
        and one frequency-boost step, keeping whichever strictly improves its cost.
        """
        cfg = self.cfg
        for i in range(cfg.elitism, len(assign)):
            if rng.random() >= cfg.local_search_rate:
                continue
            a, b = assign[i], boost[i]
            cost = objective.cost(a, b)

            util = objective.core_utilization(a)[:, -1]
            busiest, idlest = int(np.argmax(util)), int(np.argmin(util))
            members = np.flatnonzero(a == busiest)
            if busiest != idlest and members.size:
                trial = a.copy()
                trial[int(rng.choice(members))] = idlest
                moved = objective.cost(trial, b)
                if moved < cost:
                    a, cost = trial, moved

            core = int(rng.integers(0, len(b)))
            for delta in (-1, 1):
                trial = b.copy()
                trial[core] = min(max(int(trial[core]) + delta, 0), cfg.level_boost_max)
                stepped = objective.cost(a, trial)
                if stepped < cost:
                    b, cost = trial, stepped

            assign[i], boost[i] = a, b

    def _next_generation(self, objective, assign, boost, fitness, rng):
        cfg = self.cfg
        n, m = assign.shape[1], boost.shape[1]
        order = np.argsort(fitness)

        new_assign = np.empty_like(assign)
        new_boost = np.empty_like(boost)
        new_assign[: cfg.elitism] = assign[order[: cfg.elitism]]
        new_boost[: cfg.elitism] = boost[order[: cfg.elitism]]

        for slot in range(cfg.elitism, cfg.population):
            p1 = _tournament(fitness, rng, cfg.tournament)
            p2 = _tournament(fitness, rng, cfg.tournament)
            if rng.random() < cfg.crossover_rate:
                child_a = np.where(rng.random(n) < 0.5, assign[p1], assign[p2])
                child_b = np.where(rng.random(m) < 0.5, boost[p1], boost[p2])
            else:
                child_a, child_b = assign[p1].copy(), boost[p1].copy()

            mut = rng.random(n) < cfg.mutation_rate
            child_a[mut] = rng.integers(0, m, size=int(mut.sum()))
            bmut = rng.random(m) < cfg.mutation_rate
            child_b[bmut] = rng.integers(0, cfg.level_boost_max + 1, size=int(bmut.sum()))

            new_assign[slot], new_boost[slot] = child_a, child_b

        return new_assign, new_boost


def _tournament(fitness: np.ndarray, rng: np.random.Generator, k: int) -> int:
    picks = rng.integers(0, len(fitness), size=k)
    return int(picks[np.argmin(fitness[picks])])
