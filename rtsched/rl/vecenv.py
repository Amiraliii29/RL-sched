"""Padded, mask-aware vectorised environments.

Gymnasium's vector wrappers cannot carry the per-step action mask these environments need, and
they cannot batch platforms of different sizes.  Both are handled here: :class:`PaddedEnv` pads
every observation to ``max_cores`` and marks the padding invalid, so 8-, 16- and 32-core
episodes batch together and a single agent trains across all three.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable

import numpy as np

from rtsched.rl.encoding import pad_cores, pad_mask

_RESET, _STEP, _CLOSE = 0, 1, 2


class PaddedEnv:
    """Pads a scheduling environment to a fixed core count and normalises its observation keys."""

    def __init__(self, env, max_cores: int, n_sub: int, has_reject: bool, context_key: str):
        self.env = env
        self.max_cores = max_cores
        self.n_sub = n_sub
        self.has_reject = has_reject
        self.context_key = context_key
        self.n_cores = env.n_cores
        self.n_actions = max_cores * n_sub + (1 if has_reject else 0)
        if env.n_cores > max_cores:
            raise ValueError(f"{env.n_cores} cores exceeds max_cores={max_cores}")

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return self._pad(obs), self._mask()

    def step(self, action: int):
        obs, reward, done, truncated, info = self.env.step(self._translate(action))
        return self._pad(obs), self._mask(), reward, done or truncated, info

    def _translate(self, action: int) -> int:
        """Padded action index -> the wrapped environment's index."""
        if self.has_reject and action == self.n_actions - 1:
            return self.n_cores * self.n_sub
        return int(action)

    def _pad(self, obs: dict) -> dict:
        cores, valid = pad_cores(obs["cores"], self.max_cores)
        return {"cores": cores, "context": obs[self.context_key].astype(np.float32), "valid": valid}

    def _mask(self) -> np.ndarray:
        return pad_mask(
            self.env.action_masks(), self.n_cores, self.n_sub, self.max_cores, self.has_reject
        )


class MaskedVecEnv:
    """Runs ``PaddedEnv`` factories in worker processes and auto-resets on episode end."""

    def __init__(self, factories: list[Callable[[], PaddedEnv]]):
        ctx = mp.get_context("spawn")
        self.parents, self.processes = [], []
        for factory in factories:
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_worker, args=(child, factory), daemon=True)
            process.start()
            child.close()
            self.parents.append(parent)
            self.processes.append(process)
        self.n_envs = len(factories)
        self.closed = False

        probe = factories[0]()
        self.n_actions = probe.n_actions
        self.core_features = probe.reset()[0]["cores"].shape[1]
        self.context_features = probe.reset()[0]["context"].shape[0]
        self.max_cores = probe.max_cores

    def reset(self):
        for parent in self.parents:
            parent.send((_RESET, None))
        return self._gather([parent.recv() for parent in self.parents])

    def step(self, actions: np.ndarray):
        for parent, action in zip(self.parents, actions):
            parent.send((_STEP, int(action)))
        payloads = [parent.recv() for parent in self.parents]
        obs, mask = self._gather([(p[0], p[1]) for p in payloads])
        rewards = np.array([p[2] for p in payloads], dtype=np.float32)
        dones = np.array([p[3] for p in payloads], dtype=bool)
        return obs, mask, rewards, dones, [p[4] for p in payloads]

    @staticmethod
    def _gather(payloads):
        obs = {
            key: np.stack([payload[0][key] for payload in payloads])
            for key in ("cores", "context", "valid")
        }
        return obs, np.stack([payload[1] for payload in payloads])

    def close(self):
        if self.closed:
            return
        self.closed = True
        for parent in self.parents:
            parent.send((_CLOSE, None))
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def __del__(self):
        self.close()


def _worker(pipe, factory) -> None:
    env = factory()
    episode_return, episode_length = 0.0, 0
    try:
        while True:
            command, payload = pipe.recv()
            if command == _CLOSE:
                break
            if command == _RESET:
                obs, mask = env.reset()
                episode_return, episode_length = 0.0, 0
                pipe.send((obs, mask))
                continue

            obs, mask, reward, done, info = env.step(payload)
            episode_return += reward
            episode_length += 1
            if done:
                # Report the finished episode, then hand back the first observation of the next
                # one so the rollout never stalls on a terminal state.
                info = {
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    **{k: v for k, v in info.items() if k in ("feasible", "cost")},
                }
                obs, mask = env.reset()
                episode_return, episode_length = 0.0, 0
            else:
                info = {}
            pipe.send((obs, mask, reward, done, info))
    finally:
        pipe.close()
