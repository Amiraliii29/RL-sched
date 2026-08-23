"""The offline allocation objective, shared by every mapping solver.

Keeping the cost function here rather than inside a solver is what makes the phase-1 and
phase-2 comparison apples-to-apples: the genetic algorithm minimises :meth:`cost` and the
reinforcement-learning mapping environment uses ``-cost`` as its terminal reward, so both
optimise an identical trade-off between schedulability, TDP headroom, energy and the spare
capacity the slack-stealing server later turns into soft-task QoS.
"""

from __future__ import annotations

import numpy as np

from rtsched.analysis.schedulability import exec_ticks, min_feasible_level
from rtsched.config import LEVEL_DEN, GAConfig
from rtsched.model.platform import Platform
from rtsched.model.taskset import TaskSet
from rtsched.policies.base import Allocation


class AllocationObjective:
    """Vectorised evaluation of a (task -> core, per-core boost) genome."""

    def __init__(self, taskset: TaskSet, platform: Platform, weights: GAConfig | None = None):
        self.taskset = taskset
        self.platform = platform
        self.w = weights or GAConfig()
        self.tasks = taskset.periodic
        self.n_tasks = len(self.tasks)
        self.n_cores = platform.n_cores
        self.levels = np.array(platform.levels, dtype=np.float64)
        self.n_levels = len(self.levels)
        self.task_u = np.array([t.utilization for t in self.tasks])
        self.u_at_level = np.array(
            [[exec_ticks(t, int(l)) / t.period for l in platform.levels] for t in self.tasks]
        )

        power = platform.power
        volts = np.array([power.voltage(int(v)) for v in self.levels])
        self.p_active = power.cfg.static_w + power._c_eff * volts**2 * (self.levels / LEVEL_DEN)
        self.p_idle = power.cfg.static_w
        self.uncore = power.cfg.uncore_w
        self.tdp = power.tdp
        self.p_max = power.max_chip_power

    def core_utilization(self, assign: np.ndarray, placed: np.ndarray | None = None) -> np.ndarray:
        """``(n_cores, n_levels)`` per-core utilization at every frequency level.

        ``placed`` selects a subset of tasks, which lets a partially built assignment be scored
        during sequential mapping.
        """
        index = assign if placed is None else assign[placed]
        weights = self.u_at_level if placed is None else self.u_at_level[placed]
        return np.column_stack(
            [
                np.bincount(index, weights=weights[:, l], minlength=self.n_cores)
                for l in range(self.n_levels)
            ]
        )

    def resolve_levels(self, assign: np.ndarray, boost: np.ndarray, placed=None):
        util = self.core_utilization(assign, placed)
        ok = util <= 1.0 + 1e-9
        infeasible = ~ok.any(axis=1)
        base_idx = np.where(infeasible, self.n_levels - 1, ok.argmax(axis=1))
        level_idx = np.clip(base_idx + boost, 0, self.n_levels - 1)
        return util, base_idx, level_idx, infeasible

    def cost(self, assign: np.ndarray, boost: np.ndarray, placed=None) -> float:
        """Cost of a full assignment, or of the subset selected by ``placed``.

        The partial form is a potential function: shaping a sequential mapping with its
        step-to-step difference telescopes to the full cost and so leaves the optimal policy
        unchanged (Ng et al., 1999).
        """
        w = self.w
        util, _, level_idx, _ = self.resolve_levels(assign, boost, placed)
        busy = np.minimum(1.0, util[np.arange(self.n_cores), level_idx])

        overload = np.maximum(0.0, util[:, -1] - 1.0).sum()
        chip = self.uncore + (busy * self.p_active[level_idx] + (1.0 - busy) * self.p_idle).sum()
        peak = self.uncore + self.p_active[level_idx].sum()

        return float(
            w.w_infeasible * overload
            + w.w_tdp * max(0.0, chip - self.tdp) / self.tdp
            + w.w_tdp_peak * max(0.0, peak - self.tdp) / self.tdp
            + w.w_energy * chip / self.p_max
            - w.w_slack * float(np.mean(1.0 - busy))
            + w.w_balance * float(np.std(busy))
        )

    def greedy_assignment(self) -> np.ndarray:
        """Worst-fit-decreasing packing, used to seed search and as a fallback."""
        assign = np.zeros(self.n_tasks, dtype=np.int32)
        load = np.zeros(self.n_cores)
        for tid in np.argsort(-self.task_u):
            core = int(np.argmin(load))
            assign[tid] = core
            load[core] += self.task_u[tid]
        return assign

    def to_allocation(self, assign, boost, diagnostics: dict | None = None) -> Allocation:
        groups: list[list] = [[] for _ in range(self.n_cores)]
        core_of_task = {}
        for i, task in enumerate(self.tasks):
            core = int(assign[i])
            core_of_task[task.tid] = core
            groups[core].append(task)

        base_levels, feasible = [], True
        for group in groups:
            level = min_feasible_level(group, self.platform.levels)
            if level is None:
                level, feasible = self.platform.max_level, False
            base_levels.append(level)

        run_levels = tuple(
            self.platform.level_at(self.platform.level_index(l) + int(b))
            for l, b in zip(base_levels, boost)
        )
        return Allocation(
            core_of_task=core_of_task,
            level_of_core=run_levels,
            base_level_of_core=tuple(base_levels),
            feasible=feasible,
            diagnostics=diagnostics or {},
        )
