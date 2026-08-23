"""A scheduling-shaped environment with a known optimum, used to verify the learner itself.

If PPO, the masking, GAE or the core-scoring head is wrong, nothing downstream will say so --
a scheduling agent that fails to beat the baseline looks exactly like a hard problem.  This
contextual bandit removes that ambiguity: each step shows ``n_cores`` rows, the reward is the
first feature of the chosen row, and the optimal policy is simply argmax, worth ``m / (m + 1)``
in expectation against ``0.5`` for a random policy.

Lives in the package rather than the test suite because spawned workers must be able to import
it by name.
"""

from __future__ import annotations

import numpy as np

CORE_FEATURES = 4
CONTEXT_FEATURES = 3


class BestCoreEnv:
    """One-step episodes: pick the row with the largest first feature."""

    def __init__(self, n_cores: int = 6, seed: int = 0, mask_last: bool = False):
        self.n_cores = n_cores
        self.mask_last = mask_last
        self.rng = np.random.default_rng(seed)
        self._draw()

    def _draw(self) -> None:
        self.cores = self.rng.random((self.n_cores, CORE_FEATURES)).astype(np.float32)
        if self.mask_last:
            # Make the illegal action the most attractive one, so a policy that ignores the
            # mask scores visibly better in training and illegally at evaluation.
            self.cores[-1, 0] = 2.0

    def optimal(self) -> float:
        legal = self.cores[:-1, 0] if self.mask_last else self.cores[:, 0]
        return float(legal.max())

    def _obs(self) -> dict:
        return {"cores": self.cores, "task": np.zeros(CONTEXT_FEATURES, dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        self._draw()
        return self._obs(), {}

    def step(self, action: int):
        reward = float(self.cores[int(action), 0])
        self._draw()
        return self._obs(), reward, True, False, {"optimal": self.optimal()}

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.n_cores, dtype=bool)
        if self.mask_last:
            mask[-1] = False
        return mask


class BestCoreFactory:
    """Picklable factory so the diagnostic runs under the same spawned-worker path as training."""

    def __init__(self, n_cores: int = 6, seed: int = 0, max_cores: int = 8, mask_last: bool = False):
        self.n_cores = n_cores
        self.seed = seed
        self.max_cores = max_cores
        self.mask_last = mask_last

    def __call__(self):
        from rtsched.rl.vecenv import PaddedEnv

        env = BestCoreEnv(self.n_cores, self.seed, self.mask_last)
        return PaddedEnv(env, self.max_cores, n_sub=1, has_reject=False, context_key="task")
