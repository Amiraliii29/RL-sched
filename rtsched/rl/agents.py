"""Trained-agent wrappers: checkpoint format, environment factories and greedy inference.

An agent owns the two things that must stay together for a checkpoint to be usable: the network
weights and the observation contract they were trained under (core-feature width, sub-action
count, padded core limit).  Loading validates the contract instead of failing later with a shape
error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rtsched.config import ExperimentConfig
from rtsched.rl.networks import CoreScorerActorCritic, resolve_device
from rtsched.rl.vecenv import PaddedEnv

MAX_CORES = 32


@dataclass(frozen=True)
class AgentSpec:
    """Everything needed to rebuild a policy network and drive it on a fresh environment."""

    kind: str
    core_features: int
    context_features: int
    n_sub: int
    has_reject: bool
    hidden: int
    max_cores: int = MAX_CORES

    @property
    def context_key(self) -> str:
        return "task" if self.kind == "mapping" else "job"


def mapping_spec(cfg: ExperimentConfig, hidden: int = 128) -> AgentSpec:
    from rtsched.sim.features import CORE_FEATURES, MAPPING_TASK_FEATURES

    return AgentSpec(
        kind="mapping",
        core_features=CORE_FEATURES,
        context_features=MAPPING_TASK_FEATURES,
        n_sub=cfg.ga.level_boost_max + 1,
        has_reject=False,
        hidden=hidden,
    )


def online_spec(cfg: ExperimentConfig, hidden: int = 128) -> AgentSpec:
    from rtsched.sim.features import CORE_FEATURES, ONLINE_JOB_FEATURES

    return AgentSpec(
        kind="online",
        core_features=CORE_FEATURES,
        context_features=ONLINE_JOB_FEATURES,
        n_sub=len(cfg.platform.freq_levels),
        has_reject=True,
        hidden=hidden,
    )


def build_policy(spec: AgentSpec) -> CoreScorerActorCritic:
    return CoreScorerActorCritic(
        core_features=spec.core_features,
        context_features=spec.context_features,
        n_sub=spec.n_sub,
        hidden=spec.hidden,
        has_reject=spec.has_reject,
    )


def wrap(env, spec: AgentSpec) -> PaddedEnv:
    return PaddedEnv(env, spec.max_cores, spec.n_sub, spec.has_reject, spec.context_key)


def save(path: Path, policy: CoreScorerActorCritic, spec: AgentSpec, meta: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "spec": asdict(spec), "meta": meta}, path)
    return path


def load(path: Path, device: str | torch.device = "auto"):
    device = resolve_device(device) if isinstance(device, str) else device
    payload = torch.load(path, map_location=device, weights_only=False)
    spec = AgentSpec(**payload["spec"])
    policy = build_policy(spec)
    policy.load_state_dict(payload["state_dict"])
    policy.to(device).eval()
    return policy, spec, payload.get("meta", {}), device


class GreedyActor:
    """Runs a trained policy on a single padded environment.

    ``deterministic`` picks the argmax; sampling is the right mode wherever the decision is over
    interchangeable cores.  The network is permutation-equivariant by construction, so two idle
    cores receive *identical* logits and an argmax always resolves the tie the same way -- every
    soft job then queues behind the previous one on a single core while the rest idle.  Sampling
    both breaks the tie correctly and matches how the policy was trained.
    """

    def __init__(
        self,
        policy: CoreScorerActorCritic,
        spec: AgentSpec,
        device,
        deterministic: bool = True,
        seed: int = 0,
    ):
        self.policy = policy
        self.spec = spec
        self.device = device
        self.deterministic = deterministic
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def reseed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def act(self, obs: dict, mask: np.ndarray) -> int:
        with torch.no_grad():
            dist, _ = self.policy.distribution(*self._batch(obs, mask))
            if self.deterministic:
                return int(dist.probs.argmax(-1).item())
            probs = dist.probs.squeeze(0).to("cpu")
            return int(torch.multinomial(probs, 1, generator=self.generator).item())

    def _batch(self, obs: dict, mask: np.ndarray):
        to = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.device).unsqueeze(0)
        return (
            to(obs["cores"], torch.float32),
            to(obs["context"], torch.float32),
            to(obs["valid"], torch.bool),
            to(mask, torch.bool),
        )

    def rollout(self, env: PaddedEnv):
        """Drives one episode to completion and returns ``(total_reward, final_info)``."""
        obs, mask = env.reset()
        total, done, info = 0.0, False, {}
        while not done:
            obs, mask, reward, done, info = env.step(self.act(obs, mask))
            total += reward
        return total, info
