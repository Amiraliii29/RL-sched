"""Actor-critic network shared by both scheduling agents.

The policy scores every core with the *same* weights and reduces over cores with a mean pool,
which makes the logits permutation-equivariant and the value permutation-invariant.  A single
agent therefore covers 8, 16 and 32 cores: nothing in the architecture depends on how many
there are.  Compare a flat MLP over a concatenated core vector, which fixes the core count in
its first weight matrix and needs one agent per platform size.

Illegal actions are removed by adding a large negative bias before the softmax, so masked
entries get exactly zero probability and contribute nothing to the entropy term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

MASK_BIAS = -1e8


def mlp(sizes: list[int], activation=nn.Tanh, out_gain: float = 2.0**0.5) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        linear = nn.Linear(a, b)
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, gain=1.0 if last and out_gain == 1.0 else 2.0**0.5)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(activation())
    if out_gain != 2.0**0.5:
        nn.init.orthogonal_(layers[-1].weight, gain=out_gain)
    return nn.Sequential(*layers)


class CoreScorerActorCritic(nn.Module):
    """Per-core encoder + pooled context, producing ``n_cores * n_sub`` (+ optional reject) logits."""

    def __init__(
        self,
        core_features: int,
        context_features: int,
        n_sub: int,
        hidden: int = 128,
        has_reject: bool = False,
    ):
        super().__init__()
        self.n_sub = n_sub
        self.has_reject = has_reject

        self.core_encoder = mlp([core_features, hidden, hidden])
        self.context_encoder = mlp([context_features, hidden, hidden])
        self.scorer = mlp([3 * hidden, hidden, n_sub], out_gain=0.01)
        self.critic = mlp([2 * hidden, hidden, 1], out_gain=1.0)
        if has_reject:
            self.reject_head = mlp([2 * hidden, hidden, 1], out_gain=0.01)

    def forward(self, cores: torch.Tensor, context: torch.Tensor, valid: torch.Tensor):
        """``cores`` is ``(B, m, F)``; ``valid`` marks real (non-padding) cores as ``(B, m)``."""
        core_h = self.core_encoder(cores)
        ctx_h = self.context_encoder(context)

        weights = valid.unsqueeze(-1).to(core_h.dtype)
        pooled = (core_h * weights).sum(1) / weights.sum(1).clamp(min=1.0)

        joint = torch.cat(
            [core_h, ctx_h.unsqueeze(1).expand_as(core_h), pooled.unsqueeze(1).expand_as(core_h)],
            dim=-1,
        )
        logits = self.scorer(joint).flatten(1)

        summary = torch.cat([pooled, ctx_h], dim=-1)
        if self.has_reject:
            logits = torch.cat([logits, self.reject_head(summary)], dim=-1)
        return logits, self.critic(summary).squeeze(-1)

    def distribution(self, cores, context, valid, mask) -> tuple[Categorical, torch.Tensor]:
        logits, value = self.forward(cores, context, valid)
        return Categorical(logits=logits + (~mask) * MASK_BIAS), value

    @torch.no_grad()
    def act(self, cores, context, valid, mask, deterministic: bool = False):
        dist, value = self.distribution(cores, context, valid, mask)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, cores, context, valid, mask, actions):
        dist, value = self.distribution(cores, context, valid, mask)
        return dist.log_prob(actions), dist.entropy(), value


def resolve_device(preference: str = "auto") -> torch.device:
    """``auto`` uses CUDA when a working driver is present, otherwise CPU."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
