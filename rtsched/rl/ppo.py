"""Proximal Policy Optimization with invalid-action masking.

Written out rather than taken from a library for one architectural reason: the policy head must
score each core with shared weights to stay core-count agnostic, and the usual library policy
computes its logits from a pooled latent through a fixed-width linear layer, which fixes the
core count.  Everything else follows the standard recipe -- GAE, a clipped surrogate, clipped
value loss, entropy bonus, advantage normalisation per minibatch, gradient clipping and linear
schedules on both learning rate and entropy coefficient.

The masking is applied inside the distribution (see :mod:`rtsched.rl.networks`), so log-probs
and entropy are all computed on the renormalised, legal-only distribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from rtsched.rl.networks import CoreScorerActorCritic
from rtsched.rl.vecenv import MaskedVecEnv


@dataclass
class PPOConfig:
    total_steps: int = 400_000
    rollout_steps: int = 256
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    entropy_coef_final: float = 0.002
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: int = 128
    anneal: bool = True
    log_every: int = 5
    seed: int = 0


@dataclass
class TrainingLog:
    updates: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    episode_return: list[float] = field(default_factory=list)
    policy_loss: list[float] = field(default_factory=list)
    value_loss: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    approx_kl: list[float] = field(default_factory=list)

    def rows(self) -> list[dict]:
        return [
            {
                "update": u,
                "steps": s,
                "episode_return": r,
                "policy_loss": p,
                "value_loss": v,
                "entropy": e,
                "approx_kl": k,
            }
            for u, s, r, p, v, e, k in zip(
                self.updates,
                self.steps,
                self.episode_return,
                self.policy_loss,
                self.value_loss,
                self.entropy,
                self.approx_kl,
            )
        ]


class ReturnScaler:
    """Divides each environment's rewards by its own running return standard deviation.

    Two reasons the scale matters here.  The mapping objective mixes a large infeasibility
    penalty with small energy terms, so raw rewards span orders of magnitude and the value head
    cannot fit them.  And the environments are heterogeneous -- a saturated 32-core platform
    produces costs roughly forty times those of a lightly loaded 8-core one -- so a *shared*
    scale, the usual choice for identical environments, divides the low-load configurations by
    the high-load spread and erases their learning signal entirely.

    Scaling never re-centres: subtracting a mean would flip the sign of a penalty.
    """

    def __init__(self, n_envs: int, gamma: float, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = np.zeros(n_envs, dtype=np.float64)
        self.mean = np.zeros(n_envs, dtype=np.float64)
        self.m2 = np.zeros(n_envs, dtype=np.float64)
        self.count = np.zeros(n_envs, dtype=np.float64)

    @property
    def var(self) -> np.ndarray:
        return np.where(self.count > 1, self.m2 / np.maximum(self.count, 1.0), 1.0)

    def scale(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        self._update(self.returns)
        self.returns[dones] = 0.0
        return rewards / np.sqrt(self.var + self.epsilon)

    def _update(self, values: np.ndarray) -> None:
        """Per-environment Welford update, one sample each."""
        self.count += 1.0
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)


class RolloutBuffer:
    """Fixed-size on-policy storage for one rollout across all environments."""

    def __init__(self, steps: int, n_envs: int, max_cores: int, core_f: int, ctx_f: int, n_actions: int, device):
        shape = (steps, n_envs)
        self.cores = torch.zeros(*shape, max_cores, core_f, device=device)
        self.context = torch.zeros(*shape, ctx_f, device=device)
        self.valid = torch.zeros(*shape, max_cores, dtype=torch.bool, device=device)
        self.masks = torch.zeros(*shape, n_actions, dtype=torch.bool, device=device)
        self.actions = torch.zeros(*shape, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(*shape, device=device)
        self.values = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.dones = torch.zeros(*shape, device=device)
        self.steps = steps

    def add(self, t, obs, mask, action, log_prob, value, reward, done) -> None:
        self.cores[t], self.context[t], self.valid[t] = obs["cores"], obs["context"], obs["valid"]
        self.masks[t], self.actions[t], self.log_probs[t] = mask, action, log_prob
        self.values[t], self.rewards[t], self.dones[t] = value, reward, done

    def advantages(self, last_value, gamma: float, lam: float):
        """GAE(lambda).

        ``dones[t]`` flags that *transition* ``t`` ended its episode, so it is the right mask for
        the bootstrap out of ``values[t + 1]``.  Masking with ``dones[t - 1]`` instead -- the
        other common convention, which stores "state t begins an episode" -- leaks value across
        episode boundaries and is silently wrong whenever reward is concentrated at the end.
        """
        adv = torch.zeros_like(self.rewards)
        running = torch.zeros_like(last_value)
        for t in reversed(range(self.steps)):
            next_value = last_value if t == self.steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            running = delta + gamma * lam * non_terminal * running
            adv[t] = running
        return adv, adv + self.values

    def flat(self, advantages, returns):
        return {
            "cores": self.cores.flatten(0, 1),
            "context": self.context.flatten(0, 1),
            "valid": self.valid.flatten(0, 1),
            "masks": self.masks.flatten(0, 1),
            "actions": self.actions.flatten(0, 1),
            "log_probs": self.log_probs.flatten(0, 1),
            "values": self.values.flatten(0, 1),
            "advantages": advantages.flatten(0, 1),
            "returns": returns.flatten(0, 1),
        }


class PPOTrainer:
    def __init__(self, envs: MaskedVecEnv, policy: CoreScorerActorCritic, cfg: PPOConfig, device):
        self.envs = envs
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.log = TrainingLog()
        torch.manual_seed(cfg.seed)

    def train(self, progress=None) -> TrainingLog:
        cfg = self.cfg
        n_envs = self.envs.n_envs
        batch = cfg.rollout_steps * n_envs
        updates = max(1, cfg.total_steps // batch)

        buffer = RolloutBuffer(
            cfg.rollout_steps, n_envs, self.envs.max_cores, self.envs.core_features,
            self.envs.context_features, self.envs.n_actions, self.device,
        )
        obs, mask = self._to_torch(*self.envs.reset())
        scaler = ReturnScaler(n_envs, cfg.gamma)
        recent: list[float] = []
        started = time.perf_counter()

        for update in range(1, updates + 1):
            self._anneal(update, updates)
            for t in range(cfg.rollout_steps):
                with torch.no_grad():
                    action, log_prob, _, value = self.policy.act(
                        obs["cores"], obs["context"], obs["valid"], mask
                    )
                next_obs, next_mask, reward, done, infos = self.envs.step(action.cpu().numpy())
                buffer.add(
                    t, obs, mask, action, log_prob, value,
                    torch.as_tensor(scaler.scale(reward, done), device=self.device),
                    torch.as_tensor(done, dtype=torch.float32, device=self.device),
                )
                obs, mask = self._to_torch(next_obs, next_mask)
                recent += [i["episode_return"] for i in infos if "episode_return" in i]

            with torch.no_grad():
                _, last_value = self.policy(obs["cores"], obs["context"], obs["valid"])
            advantages, returns = buffer.advantages(last_value, cfg.gamma, cfg.gae_lambda)
            stats = self._optimize(buffer.flat(advantages, returns), batch)

            recent = recent[-200:]
            self.log.updates.append(update)
            self.log.steps.append(update * batch)
            self.log.episode_return.append(float(np.mean(recent)) if recent else float("nan"))
            for key, value in stats.items():
                getattr(self.log, key).append(value)

            if progress and (update % cfg.log_every == 0 or update == updates):
                progress(update, updates, self.log, time.perf_counter() - started)
        return self.log

    def _anneal(self, update: int, updates: int) -> None:
        if not self.cfg.anneal:
            return
        frac = 1.0 - (update - 1) / updates
        for group in self.optimizer.param_groups:
            group["lr"] = frac * self.cfg.learning_rate

    def _entropy_coef(self, update: int, updates: int) -> float:
        cfg = self.cfg
        frac = 1.0 - (update - 1) / max(1, updates)
        return cfg.entropy_coef_final + frac * (cfg.entropy_coef - cfg.entropy_coef_final)

    def _optimize(self, data: dict, batch: int) -> dict:
        cfg = self.cfg
        minibatch = max(1, batch // cfg.minibatches)
        index = np.arange(batch)
        entropy_coef = self._entropy_coef(len(self.log.updates) + 1, max(1, cfg.total_steps // batch))
        losses = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

        for _ in range(cfg.epochs):
            np.random.shuffle(index)
            for start in range(0, batch, minibatch):
                sub = torch.as_tensor(index[start : start + minibatch], device=self.device)
                log_prob, entropy, value = self.policy.evaluate(
                    data["cores"][sub], data["context"][sub], data["valid"][sub],
                    data["masks"][sub], data["actions"][sub],
                )
                ratio = (log_prob - data["log_probs"][sub]).exp()
                adv = data["advantages"][sub]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                policy_loss = -torch.min(
                    ratio * adv, ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * adv
                ).mean()

                clipped = data["values"][sub] + (value - data["values"][sub]).clamp(
                    -cfg.value_clip_range, cfg.value_clip_range
                )
                value_loss = 0.5 * torch.max(
                    (value - data["returns"][sub]).pow(2), (clipped - data["returns"][sub]).pow(2)
                ).mean()

                loss = policy_loss + cfg.value_coef * value_loss - entropy_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    losses["approx_kl"].append(
                        float(((ratio - 1) - (log_prob - data["log_probs"][sub])).mean())
                    )
                    losses["policy_loss"].append(float(policy_loss.detach()))
                    losses["value_loss"].append(float(value_loss.detach()))
                    losses["entropy"].append(float(entropy.mean().detach()))

        return {key: float(np.mean(values)) for key, values in losses.items()}

    def _to_torch(self, obs: dict, mask: np.ndarray):
        tensors = {
            "cores": torch.as_tensor(obs["cores"], dtype=torch.float32, device=self.device),
            "context": torch.as_tensor(obs["context"], dtype=torch.float32, device=self.device),
            "valid": torch.as_tensor(obs["valid"], device=self.device),
        }
        return tensors, torch.as_tensor(mask, device=self.device)
