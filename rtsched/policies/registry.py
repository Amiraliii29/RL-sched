"""Name -> policy factory lookup, so the runner and CLI never import policies directly.

Phase 2 registers its learned policies here and every experiment becomes reachable with
``--offline rl --online rl``; nothing else in the pipeline changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from rtsched.config import ExperimentConfig
from rtsched.policies.base import OfflinePolicy, OnlinePolicy
from rtsched.policies.offline_ffd import FirstFitPolicy
from rtsched.policies.offline_ga import GeneticMappingPolicy
from rtsched.policies.online_slack import SlackStealingPolicy

OfflineFactory = Callable[[ExperimentConfig], OfflinePolicy]
OnlineFactory = Callable[[ExperimentConfig], OnlinePolicy]

MAPPING_CHECKPOINT = Path("models/mapping.pt")
ONLINE_CHECKPOINT = Path("models/online.pt")

OFFLINE: dict[str, OfflineFactory] = {
    "ffd": lambda cfg: FirstFitPolicy(),
    "ga": lambda cfg: GeneticMappingPolicy(cfg.ga),
    "rl": lambda cfg: _rl_mapping(cfg),
}

ONLINE: dict[str, OnlineFactory] = {
    "slack": lambda cfg: SlackStealingPolicy(),
    "slack_no_dvfs": lambda cfg: SlackStealingPolicy(allow_boost=False),
    "rl": lambda cfg: _rl_online(cfg),
}


@lru_cache(maxsize=8)
def _load_mapping(checkpoint: Path, key):
    from rtsched.policies.offline_rl import RLMappingPolicy

    return RLMappingPolicy(checkpoint, _rebuild(key))


@lru_cache(maxsize=32)
def _load_online(checkpoint: Path, key, seed: int):
    from rtsched.policies.online_rl import RLOnlinePolicy

    cfg = _rebuild(key)
    return RLOnlinePolicy(checkpoint, replace(cfg, seed=seed))


def _rl_mapping(cfg: ExperimentConfig):
    _require(MAPPING_CHECKPOINT, "mapping")
    return _load_mapping(MAPPING_CHECKPOINT, _key(cfg))


def _rl_online(cfg: ExperimentConfig):
    _require(ONLINE_CHECKPOINT, "online")
    return _load_online(ONLINE_CHECKPOINT, _key(cfg), cfg.seed)


def _require(checkpoint: Path, stage: str) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found -- train it with `python -m rtsched.rl.train {stage}`"
        )


def _key(cfg: ExperimentConfig):
    """Hashable identity of the parts of the config an RL policy depends on."""
    return (cfg.platform.n_cores, cfg.platform.freq_levels, cfg.ga.level_boost_max)


def _rebuild(key) -> ExperimentConfig:
    n_cores, levels, boost = key
    from rtsched.config import GAConfig, PlatformConfig, TaskGenConfig

    return ExperimentConfig(
        platform=PlatformConfig(n_cores=n_cores, freq_levels=levels),
        taskgen=TaskGenConfig(),
        ga=GAConfig(level_boost_max=boost),
    )


def register_offline(name: str, factory: OfflineFactory) -> None:
    OFFLINE[name] = factory


def register_online(name: str, factory: OnlineFactory) -> None:
    ONLINE[name] = factory


def make_offline(name: str, cfg: ExperimentConfig) -> OfflinePolicy:
    return _make(OFFLINE, name, cfg, "offline")


def make_online(name: str, cfg: ExperimentConfig) -> OnlinePolicy:
    return _make(ONLINE, name, cfg, "online")


def _make(table: dict[str, Any], name: str, cfg: ExperimentConfig, kind: str):
    try:
        return table[name](cfg)
    except KeyError:
        raise KeyError(f"unknown {kind} policy {name!r}; available: {sorted(table)}") from None
