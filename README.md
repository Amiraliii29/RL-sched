# Power-aware real-time scheduling on a homogeneous multicore

Project 3 — scheduling hard periodic and soft aperiodic tasks under a TDP budget with per-core
DVFS.

- **Phase 1** — task generation and the complete baseline method (genetic algorithm + slack
  stealing).
- **Phase 2** — the same two decision points learned with PPO, compared against the baseline on
  held-out task sets.

| | Workload | Decision | Objective |
|---|---|---|---|
| **Offline** | hard periodic tasks | task → core, frequency level | EDF-schedulable, inside TDP, low energy |
| **Online** | soft aperiodic arrivals | accept/reject, core, frequency boost | maximise QoS without breaking a hard deadline or the TDP |

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                                          # 98 tests

# phase 1 only
.venv/bin/python -m rtsched.experiments.cli all --seeds 8           # sweep + figures (~1 min)

# phase 2: train, then compare on held-out seeds
.venv/bin/python -m rtsched.rl.train mapping --steps 1200000 --envs 18
.venv/bin/python -m rtsched.rl.train bank --offline rl --envs 18
.venv/bin/python -m rtsched.rl.train online --steps 700000 --envs 18
.venv/bin/python -m rtsched.experiments.cli all --seeds 8 --with-rl

.venv/bin/python -m rtsched.experiments.cli show 16 0.75            # one configuration
```

Outputs land in `results/`: `runs.csv` (one row per run), `tables/summary.csv` (seed-averaged
with standard deviations), `figures/*.png`, and next to every figure the CSV it was drawn from.

## Architecture

The design principle is one seam, cut in exactly one place:

> **The simulator is mechanism; policies are pluggable.** EDF dispatch, slack-stealing
> accounting, DVFS actuation and power integration live in the engine. *Which core* and *what
> frequency* is all a policy decides.

```
rtsched/
  config.py               frozen dataclasses; every knob in one place
  model/                  task.py · taskset.py · platform.py
  generation/             uunifast.py · taskgen.py
  analysis/               schedulability.py · slack.py · objective.py
  power/model.py          P = P_static + C_eff·V(f)²·f, chip TDP budget
  sim/                    engine.py · env.py · features.py · trace.py
  policies/               base.py  ← THE SEAM
                          offline_ga.py · offline_ffd.py · online_slack.py   (phase 1)
                          offline_rl.py · online_rl.py                        (phase 2)
                          scripted.py · registry.py
  metrics/collect.py      every reported number, derived from the trace
  viz/                    style.py · plots.py
  experiments/            pipeline.py · runner.py · cli.py
  rl/                     phase 2 (see below)
tests/                    98 tests
```

Two protocols in `policies/base.py` are the whole interface:

```python
class OfflinePolicy(Protocol):
    def solve(self, taskset, platform, rng) -> Allocation: ...

class OnlinePolicy(Protocol):
    def reset(self, taskset, platform, allocation) -> None: ...
    def on_arrival(self, job, obs: OnlineObservation) -> OnlineDecision: ...
```

### Design decisions worth knowing

**Integer time, integer work.** Time is in microsecond ticks. Execution demand is carried in
*work units* (`ticks × 100`) so a core at frequency level `L` consumes exactly `L` work units
per tick — exact arithmetic across DVFS changes and preemptions, no float drift.

**Bounded hyperperiod.** Periods are drawn from a semi-harmonic set `{10, 20, 25, 40, 50, 100,
200} ms`, so the hyperperiod is always 200 ms. Random periods make the LCM explode and no
configuration is simulable.

**Discrete execution time everywhere.** A task's cost at level `L` is `ceil(work_units / L)`
ticks, not the continuous `C / (L/100)`. The feasibility test, the slack analysis and the
engine all use the same discrete quantity — with the continuous form, cores near full
utilization are declared feasible and then miss deadlines in the tick-granular engine.

**Sound slack stealing.** Each core has a *floor* level fixed by the offline allocation, which
it never drops below for the whole run; the online policy may only boost above it, and only
into TDP headroom. Slack is computed at the floor — the slowest speed the core can ever reach
— so a later frequency drop can never invalidate a budget granted earlier. Slack is charged as
`ceil(work / level)` per job, mirroring exactly what the engine does when a job completes on a
tick boundary and forfeits the remainder of that tick.

**Trace-first metrics.** The engine emits only job records and a piecewise-constant power
trace. Every metric and figure derives from those, so adding a metric never means re-running
the sweep.

## The baseline (phase 1)

**Offline — memetic genetic algorithm** (`policies/offline_ga.py`). A genome is a
task-to-core assignment plus a per-core *boost*: how many frequency steps above the core's
lowest schedulable level to run. Boosting costs power but buys spare capacity, which the
slack-stealing server later converts into QoS. Population 60, 80 generations, tournament
selection, elitism 2, uniform crossover, seeded by a worst-fit-decreasing packing, plus a
local-improvement step (shed a task off the busiest core; try a boost step) — plain crossover
stalls once the task count passes a few dozen, and the memetic step recovers most of it.

The cost function lives in `analysis/objective.py`, *not* in the solver:

```
cost =  w_infeasible · Σ max(0, U_c − 1)
      + w_tdp        · max(0, P_expected − TDP)/TDP
      + w_tdp_peak   · max(0, P_peak     − TDP)/TDP
      + w_energy     · P_expected / P_max
      − w_slack      · mean spare capacity
      + w_balance    · std(core busy fraction)
```

**Online — slack-stealing dispatch** (`policies/online_slack.py`). For each core it walks
frequency levels upward from the current one and takes the lowest level at which the whole
backlog fits inside the core's stolen slack and still meets the soft deadline; among those it
prefers the most headroom and the least added power. If no core can meet the deadline the job
still goes to the best-effort core so it earns partial QoS credit.

**FFD reference** (`policies/offline_ffd.py`) — first-fit-decreasing plus lowest safe level, so
the plots exercise the multi-policy path from day one.

## Results

Grid: 8 / 16 / 32 cores × per-core utilization 0.25 / 0.5 / 0.75 / 0.85 / 0.95 / 1.0 × 8 seeds
× 3 policies = 432 runs, one hyperperiod each. Evaluation seeds are held out from RL training.

### Phase 2 vs phase 1, over the 18 configurations

| metric | RL better | equal | RL worse |
|---|---|---|---|
| energy | **16** | 0 | 2 |
| TDP violation time | **9** | 8 | 1 |
| soft-task QoS | 6 | 5 | 7 |

**The headline is the trade-off, not the QoS column.** Where the GA scores higher QoS it is
buying it by running over the thermal budget; the learned policy refuses to:

| 16 cores, u=0.85 | QoS | time above TDP | energy |
|---|---|---|---|
| GA + slack | 0.261 | **87 %** | 11.97 J |
| RL + RL | 0.144 | **0 %** | 11.70 J |

| 32 cores, u=0.85 | QoS | time above TDP | energy |
|---|---|---|---|
| GA + slack | 0.458 | **98 %** | 24.02 J |
| RL + RL | 0.152 | **0 %** | 23.04 J |

The genetic algorithm treats TDP as a weighted penalty it can trade against, so at high load it
pays the penalty and serves more soft work. The agent, whose reward comes from the simulated
trace rather than a predicted average, learns to stay inside the budget — which is the
constraint the project statement actually imposes. On the two objectives with no such ambiguity,
energy and thermal compliance, the learned policy wins outright.

Two further phase-2 results:

- **One checkpoint covers all three platform sizes.** The per-core scoring head means the agent
  trained across 8/16/32 cores serves each without retraining or reshaping.
- **Inference is ~10× faster than search.** Mapping a task set takes 0.088 s with the agent
  against 0.904 s for the genetic algorithm — the offline phase becomes cheap enough to re-run
  whenever the workload changes.

On the objective the GA itself minimises, the agent beats the GA's own greedy seed on **48/48**
instances and beats the converged GA on **23/48**, winning consistently at 16 and 32 cores where
the combinatorial space is largest (cost 10.1 vs 13.1 at 32 cores, u=0.95) and losing at 8 cores
where the search space is small enough for the GA to cover well.

### Phase 1 baseline results

**The hard-deadline guarantee holds: 0 violations in 432 runs, for every policy including the
learned one.** Whenever the offline phase declares an allocation feasible, no hard job ever
misses, regardless of soft traffic — the guarantee is a property of the slack-stealing mechanism
in the engine, not of the policy on top of it, so a learned policy cannot break it.

| | GA + slack | FFD + slack |
|---|---|---|
| QoS at 8 cores, u=0.75 | **0.997** | 0.854 |
| Energy at 8 cores, u=0.75 | **5.77 J** | 6.20 J |
| TDP violation time at 8 cores, u=0.75 | **0 %** | 89 % |

Three findings the report should lead with:

1. **The GA dominates FFD outright up to u ≈ 0.75** — better QoS, less energy, and it is the
   only one that stays inside the TDP.
2. **Above u ≈ 0.8 the platform is thermally over-subscribed.** Every core's lowest feasible
   level is already ≥ 90 %, so no mapping keeps chip power under the TDP; the two solvers land
   on different points of a QoS-versus-power Pareto front rather than one dominating. This is
   the regime where a policy that can *see the trace* has something to gain — the phase-2
   opening.
3. **At exactly u = 1.0 partitioned EDF fails by construction** (schedulability drops to 0):
   packing tasks summing to *m* into *m* unit-capacity cores needs a perfect packing.
   Figure 10 shows where the knee really lives — it is driven by task *granularity*, not
   utilization alone.

### Figures

| # | File | Requirement |
|---|------|-------------|
| 1 | `01_qos_vs_utilization.png` | QoS of soft tasks |
| 2 | `02_qos_by_state.png` | QoS by system state (on time / late / rejected) |
| 3 | `03_schedulability.png` | task-set schedulability |
| 4 | `04_deadline_miss_ratio.png` | deadline miss ratio, hard and soft |
| 5 | `05_energy.png` | energy per hyperperiod |
| 6 | `06_core_power_*.png` | power consumed by each core |
| 7 | `07_power_trace_*.png` | instantaneous power per core over the hyperperiod |
| 8 | `08_task_table.png` | table of tasks and their characteristics |
| 9 | `09_ga_convergence.png` | GA convergence (supplementary) |
| 10 | `10_schedulability_granularity.png` | schedulability vs task granularity (supplementary) |
| 11 | `11_rl_training.png` | PPO learning curves, both agents (phase 2) |

## Phase 2: the learned policies

Both decision points are learned with **PPO and invalid-action masking**, behind the same two
protocols the baseline implements — so the runner, the metrics and every figure treat a learned
policy exactly like the genetic algorithm.

```
rtsched/rl/
  networks.py   per-core scorer actor-critic, masked categorical head
  ppo.py        PPO: GAE, clipped surrogate + value, entropy/LR schedules, return scaling
  vecenv.py     padded, mask-aware vectorised envs over spawned workers
  encoding.py   the one place observations and masks are padded (train == serve)
  envs.py       picklable env factories, allocation bank
  agents.py     checkpoint format, greedy inference
  toy.py        diagnostic env with a known optimum
  train.py      three-stage CLI
rtsched/policies/offline_rl.py   RLMappingPolicy  (OfflinePolicy)
rtsched/policies/online_rl.py    RLOnlinePolicy   (OnlinePolicy)
```

### The architecture, and why it is this one

The policy scores **every core with the same weights** and reduces over cores with a masked mean
pool. Logits are permutation-equivariant, the value is permutation-invariant, and nothing in the
network depends on how many cores there are — so **one checkpoint serves 8, 16 and 32 cores**. A
flat MLP over a concatenated core vector fixes the core count in its first weight matrix and
would need three separate agents, making the comparison against a single GA baseline muddier.

This is also why PPO is written out rather than imported: a stock library policy computes its
logits from a pooled latent through a fixed-width linear layer, which reintroduces exactly the
dependence on core count that the architecture removes.

### Three things that decided whether it learned at all

Each of these produced a flat learning curve until fixed; they are worth knowing before touching
the trainer.

1. **The mapping episode is two phases, not one.** Placing a task and setting its core's
   frequency in the same action means only the *last* task to land on a core determines its
   boost — the DVFS decision gets no usable credit. The episode now places all tasks, then walks
   the cores setting each boost. Before the split the agent picked boost 0 everywhere.
2. **Reward scaling is per environment.** The objective mixes a large infeasibility penalty with
   small energy terms, and a saturated 32-core platform produces costs ~40× a lightly loaded
   8-core one. A single shared scale — the usual choice — divides the low-load configurations by
   the high-load spread and erases their signal.
3. **GAE must mask with the done flag of the transition, not the previous one.** The off-by-one
   leaks value across episode boundaries and is silently wrong precisely when reward is
   concentrated at the end, as it is here.

### Training

```bash
python -m rtsched.rl.train mapping --steps 1200000 --envs 18   # stage 1
python -m rtsched.rl.train bank --offline rl                   # stage 2
python -m rtsched.rl.train online --steps 600000 --envs 18     # stage 3
```

Stage 2 exists because `OnlineEnv.reset` would otherwise re-solve the offline mapping every
episode, costing more than the episode itself; it also means the online agent trains against the
allocations it will actually meet at evaluation.

**Training instances are seeds 1000+, evaluation instances are seeds 0–7 — disjoint.** The
comparison therefore measures generalisation to unseen task sets, not memorisation. The GA
re-solves every instance from scratch and has no such split, which if anything favours it.

## Appendix: how the phase-2 seam was designed

The environments existed and were validated before any learning code was written. `tests/test_env.py` asserts that driving
`OnlineEnv` with the phase-1 heuristic reproduces `simulate()` **job for job** — so the
environment is debugged before any learning code is written, and an agent's results are
directly comparable to the baseline on identical episodes.

- `sim/env.py::MappingEnv` — offline MDP, one step per task. Terminal reward is
  `-AllocationObjective.cost`, the exact quantity the GA minimises.
- `sim/env.py::OnlineEnv` — online semi-MDP, one step per soft arrival. Reward is credited
  between decisions: QoS of jobs that completed, minus hard-miss and over-TDP penalties.
- Both expose `action_masks()` for masked policy gradients (`sb3-contrib` `MaskablePPO`).
- Observations are `(n_cores, 8)` matrices of ratios (`sim/features.py`). Score each row with a
  shared MLP and reduce over rows and **one agent covers 8, 16 and 32 cores** — no retraining
  per core count, no reshaping.

Adding phase 2 turned out to be exactly the three files the seam predicted --
`policies/offline_rl.py`, `policies/online_rl.py`, and two `registry` entries. The engine, the
generator, the metrics and every plotting function are byte-for-byte unchanged from phase 1;
`--with-rl` on the sweep is the only new flag.

### Where to train

This is a **CPU-bound** workload: the networks are small MLPs and >95 % of wall-clock is the
discrete-event simulator, so throughput is set by how many environments run in parallel, not by
GPU. On the development machine (i7-12700H, 14 cores) `SubprocVecEnv` with 12 workers gives
roughly 6–8× the environment throughput of a 2-vCPU Colab session; an RTX 3060 would be slower
than the CPU for networks this small. Train locally, cap workers at ~12 to avoid thermal
throttling, and keep the simulator fast (it is event-driven and integer-timed for this reason —
288 runs finish in ~50 s on 12 workers).

## Testing

```bash
.venv/bin/python -m pytest -q               # 98 tests
.venv/bin/python -m pytest -q -m slow       # PPO convergence diagnostics
```

The suite covers UUniFast properties and generator determinism, the EDF and demand-bound tests,
slack correctness including the per-job rounding, and — most importantly — the behavioural
guarantees: feasible allocations never miss a hard deadline, cores never run below their
allocated floor, energy reconciles with the recorded power trace, the trace tiles the
hyperperiod without gaps, and the RL environments reproduce the scripted path exactly.

Phase 2 adds architecture invariants (permutation equivariance of the logits, invariance of the
value, padding-independence, zero probability on masked actions), the reward-shaping identity,
train/serve encoding equality, and — marked `slow` — two diagnostics that check **the learner
itself** on `rtsched/rl/toy.py`, an environment whose optimum is known analytically. One asserts
PPO approaches it; the other makes the illegal action the most rewarding and asserts the agent
never takes it. Without these, a learned policy that merely fails to beat the baseline is
indistinguishable from a broken trainer.
