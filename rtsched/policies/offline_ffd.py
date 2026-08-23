"""First-Fit-Decreasing partitioning: a cheap reference point for the genetic algorithm."""

from __future__ import annotations

import numpy as np

from rtsched.analysis.schedulability import core_utilization, edf_feasible, min_feasible_level
from rtsched.model.platform import Platform
from rtsched.model.taskset import TaskSet
from rtsched.policies.base import Allocation


class FirstFitPolicy:
    """Packs tasks by decreasing utilization, then drops each core to its lowest safe level."""

    name = "ffd"

    def solve(self, taskset: TaskSet, platform: Platform, rng: np.random.Generator) -> Allocation:
        groups: list[list] = [[] for _ in range(platform.n_cores)]
        core_of_task: dict[int, int] = {}

        for task in sorted(taskset.periodic, key=lambda t: -t.utilization):
            placed = False
            for index, group in enumerate(groups):
                if edf_feasible(group + [task], platform.max_level):
                    group.append(task)
                    core_of_task[task.tid] = index
                    placed = True
                    break
            if not placed:
                index = min(range(platform.n_cores), key=lambda i: core_utilization(groups[i], platform.max_level))
                groups[index].append(task)
                core_of_task[task.tid] = index

        base_levels, feasible = [], True
        for group in groups:
            level = min_feasible_level(group, platform.levels)
            if level is None:
                level, feasible = platform.max_level, False
            base_levels.append(level)

        return Allocation(
            core_of_task=core_of_task,
            level_of_core=tuple(base_levels),
            base_level_of_core=tuple(base_levels),
            feasible=feasible,
            diagnostics={"solver": self.name},
        )
