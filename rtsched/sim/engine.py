"""Event-driven simulator for partitioned EDF with a slack-stealing aperiodic server.

Mechanism only.  Per core the engine runs the earliest-deadline hard job, except that a
pending soft job pre-empts it for as long as the core's slack allows -- which is what makes
hard deadline misses impossible on a feasible allocation.

Two invariants keep the slack account sound.  Each core has a *floor* level, fixed by the
offline allocation, that it never drops below for the whole run; the online policy may only
boost above it, and only into TDP headroom.  Slack is then computed at the floor -- the
slowest speed the core can ever run at -- so a later frequency drop can never invalidate a
budget that was granted earlier.
"""

from __future__ import annotations

import heapq

from rtsched.analysis.slack import available_slack, ceil_div
from rtsched.config import LEVEL_DEN, SECONDS_PER_TICK, SimConfig
from rtsched.model.platform import Platform
from rtsched.model.task import Job, JobKind, JobRecord, PeriodicTask
from rtsched.model.taskset import TaskSet
from rtsched.policies.base import (
    Allocation,
    CoreObservation,
    OnlineObservation,
    OnlinePolicy,
)
from rtsched.sim.trace import PowerTrace, SimResult

_RELEASE, _ARRIVAL = 0, 1


class _Core:
    __slots__ = (
        "index", "level", "floor", "base_level", "tasks", "next_release", "max_period",
        "ready", "soft", "current", "current_is_soft", "slack_budget",
        "busy_ticks", "energy_j",
    )

    def __init__(self, index: int, base_level: int, floor: int, tasks: list[PeriodicTask]):
        self.index = index
        self.base_level = base_level
        self.floor = max(floor, base_level)
        self.level = self.floor
        self.tasks = tasks
        self.next_release = {t.tid: 0 for t in tasks}
        self.max_period = max((t.period for t in tasks), default=1)
        self.ready: list[tuple[int, int, Job]] = []
        self.soft: list[tuple[int, int, Job]] = []
        self.current: Job | None = None
        self.current_is_soft = False
        self.slack_budget = 0
        self.busy_ticks = 0
        self.energy_j = 0.0

    def active_hard(self):
        return (entry[2] for entry in self.ready)


class Engine:
    """Simulates one hyperperiod of a given allocation under a given online policy."""

    def __init__(
        self,
        taskset: TaskSet,
        platform: Platform,
        allocation: Allocation,
        online_policy: OnlinePolicy,
        sim_cfg: SimConfig | None = None,
    ):
        self.taskset = taskset
        self.platform = platform
        self.allocation = allocation
        self.policy = online_policy
        self.cfg = sim_cfg or SimConfig()
        self.power = platform.power
        self.horizon = taskset.hyperperiod

        by_id = taskset.by_id()
        self.cores = [
            _Core(
                i,
                allocation.base_level_of_core[i],
                allocation.level_of_core[i],
                [by_id[tid] for tid in sorted(allocation.tasks_on(i))],
            )
            for i in range(platform.n_cores)
        ]

        self._events: list[tuple[int, int, int, object]] = []
        self._jid = 0
        self._order = 0
        self._jobs: list[Job] = []
        self.completed_soft: list[Job] = []
        self.hard_misses = 0
        self.tdp_violation_ticks = 0
        self.trace = PowerTrace() if self.cfg.record_power_trace else None

        self._seed_events()

    # ---------------------------------------------------------------- setup

    def _seed_events(self) -> None:
        for task in self.taskset.periodic:
            core = self.allocation.core_of_task[task.tid]
            for release in range(0, self.horizon, task.period):
                self._push(release, _RELEASE, (task, core))
        for soft in self.taskset.aperiodic:
            if soft.arrival < self.horizon:
                self._push(soft.arrival, _ARRIVAL, soft)

    def _push(self, time: int, kind: int, payload) -> None:
        heapq.heappush(self._events, (time, kind, self._order, payload))
        self._order += 1

    def _new_job(self, source_id, kind, release, abs_deadline, work, weight=1.0) -> Job:
        job = Job(
            jid=self._jid,
            source_id=source_id,
            kind=kind,
            release=release,
            abs_deadline=abs_deadline,
            work=work,
            remaining=work,
            weight=weight,
        )
        self._jid += 1
        self._jobs.append(job)
        return job

    # ----------------------------------------------------------------- run

    def run(self) -> SimResult:
        """Drive the simulation with ``self.policy`` making every soft-arrival decision."""
        loop = self.iter_decisions()
        try:
            job, obs = next(loop)
            while True:
                job, obs = loop.send(self.policy.on_arrival(job, obs))
        except StopIteration as stop:
            return stop.value

    def iter_decisions(self):
        """Generator form of the run loop: yields ``(job, observation)`` at every soft arrival
        and expects an :class:`OnlineDecision` back.  Returns the :class:`SimResult`.

        The reinforcement-learning environment drives this directly, so agent and scripted
        policy share one loop and cannot diverge.
        """
        now = 0
        energy = 0.0
        peak = 0.0
        steps = 0

        while now < self.horizon:
            yield from self._drain_events(now)
            for core in self.cores:
                self._select(core, now)

            nxt = self._next_boundary(now)
            span = nxt - now

            levels = [c.level for c in self.cores]
            active = [c.current is not None for c in self.cores]
            chip_w = self.power.chip_power(levels, active)

            energy += chip_w * span * SECONDS_PER_TICK
            peak = max(peak, chip_w)
            if chip_w > self.power.tdp + 1e-9:
                self.tdp_violation_ticks += span

            if self.trace is not None:
                self.trace.add_chip(now, nxt, chip_w)
                for core, act in zip(self.cores, active):
                    self.trace.add_core(
                        core.index, now, nxt, core.level, self.power.core_power(core.level, act)
                    )

            self._advance(now, nxt, span)
            now = nxt
            steps += 1

        return self._finalize(energy, peak, steps)

    # -------------------------------------------------------------- events

    def _drain_events(self, now: int):
        while self._events and self._events[0][0] <= now:
            _, kind, _, payload = heapq.heappop(self._events)
            if kind == _RELEASE:
                self._release_hard(now, *payload)
                continue

            job = self._new_job(
                payload.aid, JobKind.SOFT, now, payload.deadline, payload.work_units, payload.weight
            )
            decision = yield job, self._observe(now, job)
            self._dispatch_soft(job, decision)

    def _release_hard(self, now: int, task: PeriodicTask, core_index: int) -> None:
        core = self.cores[core_index]
        job = self._new_job(task.tid, JobKind.HARD, now, now + task.deadline, task.work_units)
        job.core = core_index
        heapq.heappush(core.ready, (job.abs_deadline, job.jid, job))
        core.next_release[task.tid] = now + task.period

    def _dispatch_soft(self, job: Job, decision) -> None:
        if decision is None or not decision.accept or not 0 <= decision.core < len(self.cores):
            job.accepted = False
            return

        core = self.cores[decision.core]
        job.core = core.index
        if decision.level >= 0:
            core.level = self._tdp_clamp(core, max(decision.level, core.floor))
        heapq.heappush(core.soft, (job.abs_deadline, job.jid, job))

    def _tdp_clamp(self, core: _Core, requested: int) -> int:
        """Highest level in ``[floor, requested]`` whose worst-case chip power fits the TDP."""
        levels = [c.level for c in self.cores]
        active = [True] * len(self.cores)
        for candidate in reversed(self.platform.levels):
            if candidate > requested or candidate < core.floor:
                continue
            levels[core.index] = candidate
            if self.power.chip_power(levels, active) <= self.power.tdp + 1e-9:
                return candidate
        return core.floor

    # ---------------------------------------------------------- scheduling

    def _slack(self, core: _Core, now: int) -> int:
        horizon = self.horizon
        if self.cfg.slack_horizon_factor:
            horizon = min(horizon, now + self.cfg.slack_horizon_factor * core.max_period)
        return available_slack(
            core.active_hard(), core.tasks, core.next_release, now, horizon, core.floor
        )

    def _select(self, core: _Core, now: int) -> None:
        core.current = None
        core.current_is_soft = False
        core.slack_budget = 0

        if core.soft:
            slack = self._slack(core, now)
            if slack > 0:
                job = core.soft[0][2]
                core.current, core.current_is_soft, core.slack_budget = job, True, slack
                if job.start < 0:
                    job.start = now
                return

        if core.ready:
            job = core.ready[0][2]
            core.current = job
            if job.start < 0:
                job.start = now

    def _next_boundary(self, now: int) -> int:
        nxt = self.horizon
        if self._events:
            nxt = min(nxt, self._events[0][0])
        for core in self.cores:
            if core.current is None:
                continue
            nxt = min(nxt, now + ceil_div(core.current.remaining, core.level))
            if core.current_is_soft:
                nxt = min(nxt, now + core.slack_budget)
        return max(nxt, now + 1)

    def _advance(self, now: int, nxt: int, span: int) -> None:
        for core in self.cores:
            running = core.current is not None
            core.energy_j += self.power.core_power(core.level, running) * span * SECONDS_PER_TICK
            if not running:
                continue

            core.busy_ticks += span
            job = core.current
            job.remaining -= span * core.level
            if job.remaining <= 0:
                job.remaining = 0
                job.finish = nxt
                heap = core.soft if core.current_is_soft else core.ready
                heapq.heappop(heap)
                if core.current_is_soft:
                    self.completed_soft.append(job)
                    if not core.soft:
                        core.level = core.floor
                elif job.finish > job.abs_deadline:
                    self.hard_misses += 1

    # ------------------------------------------------------------ policies

    def _observe(self, now: int, job: Job) -> OnlineObservation:
        levels = [c.level for c in self.cores]
        active = [c.current is not None for c in self.cores]
        cores = tuple(
            CoreObservation(
                index=c.index,
                level=c.level,
                base_level=c.floor,
                max_level=self.platform.max_level,
                utilization=sum(t.utilization for t in c.tasks),
                slack_ticks=self._slack(c, now),
                soft_backlog_ticks=sum(e[2].remaining for e in c.soft) // LEVEL_DEN,
                n_soft_pending=len(c.soft),
                busy=c.current is not None,
            )
            for c in self.cores
        )
        return OnlineObservation(
            now=now,
            horizon=self.horizon,
            cores=cores,
            chip_power_w=self.power.chip_power(levels, active),
            tdp_w=self.power.tdp,
            job_work_ticks=job.work // 100,
            job_deadline=job.abs_deadline,
            job_weight=job.weight,
        )

    # ------------------------------------------------------------- results

    def _finalize(self, energy: float, peak: float, steps: int) -> SimResult:
        duration_s = self.horizon * SECONDS_PER_TICK
        return SimResult(
            horizon=self.horizon,
            jobs=[JobRecord.of(j) for j in self._jobs],
            energy_j=energy,
            peak_chip_power_w=peak,
            mean_chip_power_w=energy / duration_s if duration_s else 0.0,
            tdp_w=self.power.tdp,
            tdp_violation_ratio=self.tdp_violation_ticks / self.horizon,
            core_energy_j=[c.energy_j for c in self.cores],
            core_busy_ratio=[c.busy_ticks / self.horizon for c in self.cores],
            core_levels=[c.level for c in self.cores],
            steps=steps,
            trace=self.trace,
        )


def simulate(
    taskset: TaskSet,
    platform: Platform,
    allocation: Allocation,
    online_policy: OnlinePolicy,
    sim_cfg: SimConfig | None = None,
) -> SimResult:
    online_policy.reset(taskset, platform, allocation)
    return Engine(taskset, platform, allocation, online_policy, sim_cfg).run()
