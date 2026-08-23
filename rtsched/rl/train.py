"""Training entry point for both agents.

    python -m rtsched.rl.train mapping                # stage 1
    python -m rtsched.rl.train bank --offline rl      # stage 2: allocations from the trained agent
    python -m rtsched.rl.train online                 # stage 3
    python -m rtsched.rl.train all

The stages are ordered so the online agent trains against the allocations it will actually meet
at evaluation time.  Training instances (seeds 1000+) are disjoint from the evaluation seeds, so
the comparison against the genetic-algorithm baseline measures generalisation to unseen task
sets, not memorisation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from rtsched.experiments.pipeline import build_config
from rtsched.rl import agents
from rtsched.rl.envs import AllocationBank, bank_points, build_factories
from rtsched.rl.networks import resolve_device
from rtsched.rl.ppo import PPOConfig, PPOTrainer
from rtsched.rl.vecenv import MaskedVecEnv

MODELS = Path("models")
LOGS = Path("results/rl")
MAPPING_CKPT = MODELS / "mapping.pt"
ONLINE_CKPT = MODELS / "online.pt"
BANK_PATH = MODELS / "allocations.pkl"


def _default_workers() -> int:
    return max(2, min(12, (os.cpu_count() or 4) - 2))


def _progress(tag: str):
    def report(update, updates, log, elapsed):
        print(
            f"\r  {tag} {update:>4}/{updates}  return {log.episode_return[-1]:+.3f}"
            f"  entropy {log.entropy[-1]:.3f}  kl {log.approx_kl[-1]:+.4f}"
            f"  {elapsed:6.1f}s",
            end="",
            flush=True,
        )
        if update == updates:
            print()

    return report


def _train(kind: str, spec, factories, cfg: PPOConfig, device, checkpoint: Path) -> None:
    envs = MaskedVecEnv(factories)
    try:
        policy = agents.build_policy(spec)
        params = sum(p.numel() for p in policy.parameters())
        print(f"{kind}: {envs.n_envs} envs, {params:,} parameters, device={device}")
        trainer = PPOTrainer(envs, policy, cfg, device)
        log = trainer.train(progress=_progress(kind))
    finally:
        envs.close()

    LOGS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log.rows()).to_csv(LOGS / f"{kind}_training.csv", index=False)
    agents.save(
        checkpoint,
        trainer.policy,
        spec,
        {"run": kind, "config": vars(cfg), "final_return": log.episode_return[-1]},
    )
    print(f"  saved {checkpoint}")


def cmd_mapping(args) -> None:
    cfg = build_config(8, 0.5, seed=0)
    spec = agents.mapping_spec(cfg, hidden=args.hidden)
    ppo = PPOConfig(
        total_steps=args.steps,
        rollout_steps=args.rollout,
        gamma=1.0,  # finite episode with a shaped potential; no discounting needed
        entropy_coef=0.02,
        learning_rate=args.lr,
        hidden=args.hidden,
        seed=args.seed,
    )
    _train("mapping", spec, build_factories("mapping", spec, args.envs), ppo, resolve_device(args.device), MAPPING_CKPT)


def cmd_bank(args) -> None:
    workers = args.workers or _default_workers()
    points = bank_points(args.envs)
    print(f"allocation bank: {len(points)} allocations from the {args.offline} solver on {workers} workers")

    def report(done, total):
        print(f"\r  {done}/{total}", end="", flush=True)
        if done == total:
            print()

    bank = AllocationBank.build(args.offline, points, workers, report)
    bank.save(BANK_PATH)
    print(f"  saved {BANK_PATH} ({len(bank)} allocations)")


def cmd_online(args) -> None:
    if not BANK_PATH.exists():
        raise SystemExit(f"{BANK_PATH} not found -- run `train bank` first")
    bank = AllocationBank.load(BANK_PATH)

    cfg = build_config(8, 0.5, seed=0)
    spec = agents.online_spec(cfg, hidden=args.hidden)
    ppo = PPOConfig(
        total_steps=args.steps,
        rollout_steps=args.rollout,
        gamma=0.99,
        entropy_coef=0.03,
        learning_rate=args.lr,
        hidden=args.hidden,
        seed=args.seed,
    )
    factories = build_factories("online", spec, args.envs, bank)
    _train("online", spec, factories, ppo, resolve_device(args.device), ONLINE_CKPT)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="rtsched.rl.train")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, steps: int):
        p.add_argument("--steps", type=int, default=steps)
        p.add_argument("--rollout", type=int, default=128)
        p.add_argument("--envs", type=int, default=_default_workers())
        p.add_argument("--hidden", type=int, default=128)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="auto")
        return p

    common(sub.add_parser("mapping"), 400_000).set_defaults(func=cmd_mapping)
    common(sub.add_parser("online"), 300_000).set_defaults(func=cmd_online)

    p = sub.add_parser("bank")
    p.add_argument("--offline", default="rl")
    p.add_argument("--envs", type=int, default=_default_workers(),
                   help="must match the worker count used for `train online`")
    p.add_argument("--workers", type=int, default=0)
    p.set_defaults(func=cmd_bank)

    p = sub.add_parser("all")
    common(p, 0)
    p.add_argument("--offline", default="rl")
    p.set_defaults(func=_run_all)

    args = parser.parse_args(argv)
    args.func(args)


def _run_all(args) -> None:
    from copy import copy

    stage = copy(args)
    stage.steps = args.steps or 400_000
    cmd_mapping(stage)
    cmd_bank(args)
    stage = copy(args)
    stage.steps = args.steps or 300_000
    cmd_online(stage)


if __name__ == "__main__":
    main()
