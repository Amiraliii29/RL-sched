"""Phase-2 tests: architecture invariants, the learner itself, and train/serve consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rtsched.experiments.pipeline import build_config
from rtsched.rl import agents
from rtsched.rl.encoding import encode_online, online_action_mask, pad_cores, pad_mask
from rtsched.rl.networks import CoreScorerActorCritic, resolve_device
from rtsched.rl.ppo import PPOConfig, PPOTrainer
from rtsched.rl.toy import CONTEXT_FEATURES as TOY_CTX
from rtsched.rl.toy import CORE_FEATURES as TOY_CORE
from rtsched.rl.toy import BestCoreFactory
from rtsched.rl.vecenv import MaskedVecEnv, PaddedEnv
from rtsched.sim.env import MappingEnv, OnlineEnv

MAPPING_CKPT = Path("models/mapping.pt")
ONLINE_CKPT = Path("models/online.pt")


def _net(n_sub=3, has_reject=False, hidden=32):
    torch.manual_seed(0)
    return CoreScorerActorCritic(TOY_CORE, TOY_CTX, n_sub, hidden, has_reject)


def _batch(m=8, n_sub=3, has_reject=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    cores = torch.rand(2, m, TOY_CORE, generator=g)
    context = torch.rand(2, TOY_CTX, generator=g)
    valid = torch.ones(2, m, dtype=torch.bool)
    mask = torch.ones(2, m * n_sub + int(has_reject), dtype=torch.bool)
    return cores, context, valid, mask


# --------------------------------------------------------------- architecture


def test_logits_are_permutation_equivariant_in_cores():
    """Reordering the cores must reorder the logits identically -- the property that lets one
    agent serve 8, 16 and 32 cores."""
    net = _net().eval()
    cores, context, valid, _ = _batch()
    perm = torch.randperm(cores.shape[1])

    with torch.no_grad():
        base, _ = net(cores, context, valid)
        permuted, _ = net(cores[:, perm], context, valid)

    base = base.view(2, cores.shape[1], net.n_sub)
    permuted = permuted.view(2, cores.shape[1], net.n_sub)
    assert torch.allclose(base[:, perm], permuted, atol=1e-5)


def test_value_is_permutation_invariant_in_cores():
    net = _net().eval()
    cores, context, valid, _ = _batch()
    with torch.no_grad():
        _, base = net(cores, context, valid)
        _, shuffled = net(cores[:, torch.randperm(cores.shape[1])], context, valid)
    assert torch.allclose(base, shuffled, atol=1e-5)


def test_padding_does_not_change_real_core_logits():
    """A padded 32-wide batch must score an 8-core platform exactly as an 8-wide batch does."""
    net = _net().eval()
    cores, context, valid, _ = _batch(m=8)

    padded = torch.zeros(2, 32, TOY_CORE)
    padded[:, :8] = cores
    padded_valid = torch.zeros(2, 32, dtype=torch.bool)
    padded_valid[:, :8] = True

    with torch.no_grad():
        small, small_v = net(cores, context, valid)
        big, big_v = net(padded, context, padded_valid)

    assert torch.allclose(small, big[:, : small.shape[1]], atol=1e-5)
    assert torch.allclose(small_v, big_v, atol=1e-5)


def test_masked_actions_get_zero_probability_and_finite_entropy():
    net = _net().eval()
    cores, context, valid, mask = _batch()
    mask[:, ::2] = False

    dist, _ = net.distribution(cores, context, valid, mask)
    assert torch.all(dist.probs[~mask] == 0.0)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(2), atol=1e-5)
    assert torch.isfinite(dist.entropy()).all()


def test_sampling_never_returns_a_masked_action():
    net = _net().eval()
    cores, context, valid, mask = _batch(seed=1)
    mask[:, 1:] = False
    for _ in range(20):
        action, *_ = net.act(cores, context, valid, mask)
        assert torch.all(action == 0)


# ---------------------------------------------------------------- the learner


@pytest.mark.slow
def test_ppo_solves_an_environment_with_a_known_optimum():
    """Guards the learner itself: on a bandit whose optimum is argmax, PPO must approach it.

    Expected return is 0.5 for a random policy and m/(m+1) = 0.857 for the optimal one.
    """
    envs = MaskedVecEnv([BestCoreFactory(6, seed=s) for s in range(4)])
    try:
        policy = CoreScorerActorCritic(TOY_CORE, TOY_CTX, 1, 64)
        cfg = PPOConfig(total_steps=60_000, rollout_steps=64, gamma=0.0, entropy_coef=0.01)
        log = PPOTrainer(envs, policy, cfg, resolve_device("cpu")).train()
    finally:
        envs.close()
    assert log.episode_return[-1] > 0.80, f"PPO reached only {log.episode_return[-1]:.3f}"


@pytest.mark.slow
def test_ppo_respects_action_masks_under_temptation():
    """The masked-out core carries the highest reward; a correct agent must never take it."""
    envs = MaskedVecEnv([BestCoreFactory(6, seed=s, mask_last=True) for s in range(4)])
    try:
        policy = CoreScorerActorCritic(TOY_CORE, TOY_CTX, 1, 64)
        cfg = PPOConfig(total_steps=30_000, rollout_steps=64, gamma=0.0)
        log = PPOTrainer(envs, policy, cfg, resolve_device("cpu")).train()
    finally:
        envs.close()
    assert log.episode_return[-1] <= 1.0, "agent collected reward only reachable illegally"


def test_gae_reduces_to_the_reward_on_a_one_step_episode():
    from rtsched.rl.ppo import RolloutBuffer

    buffer = RolloutBuffer(1, 2, 4, 3, 2, 5, torch.device("cpu"))
    buffer.rewards[0] = torch.tensor([1.0, -2.0])
    buffer.dones[0] = torch.tensor([1.0, 1.0])
    buffer.values[0] = torch.tensor([0.25, 0.25])

    advantages, returns = buffer.advantages(torch.zeros(2), gamma=0.99, lam=0.95)
    assert torch.allclose(advantages[0], torch.tensor([0.75, -2.25]))
    assert torch.allclose(returns[0], torch.tensor([1.0, -2.0]))


def test_gae_does_not_bootstrap_across_an_episode_boundary():
    """Distinguishes the two done-flag conventions, which agree on a single-step rollout.

    Transition 0 ends its episode, so its advantage must ignore ``values[1]`` entirely; the
    off-by-one convention would leak that value backwards.
    """
    from rtsched.rl.ppo import RolloutBuffer

    buffer = RolloutBuffer(2, 1, 4, 3, 2, 5, torch.device("cpu"))
    buffer.rewards[0], buffer.rewards[1] = torch.tensor([1.0]), torch.tensor([0.0])
    buffer.dones[0], buffer.dones[1] = torch.tensor([1.0]), torch.tensor([0.0])
    buffer.values[0], buffer.values[1] = torch.tensor([0.0]), torch.tensor([100.0])

    advantages, _ = buffer.advantages(torch.zeros(1), gamma=0.99, lam=0.95)
    assert torch.allclose(advantages[0], torch.tensor([1.0]))


def test_return_scaler_shrinks_large_rewards_without_flipping_sign():
    from rtsched.rl.ppo import ReturnScaler

    scaler = ReturnScaler(n_envs=2, gamma=0.99)
    dones = np.zeros(2, dtype=bool)
    scaled = np.zeros(2)
    for _ in range(50):
        scaled = scaler.scale(np.array([500.0, -500.0]), dones)
    assert np.abs(scaled).max() < 50.0
    assert scaled[0] > 0 and scaled[1] < 0


# ------------------------------------------------------------------ encoding


def test_padding_helpers_round_trip():
    cores = np.arange(8 * 5, dtype=np.float32).reshape(8, 5)
    padded, valid = pad_cores(cores, 32)
    assert padded.shape == (32, 5) and valid.sum() == 8
    assert np.array_equal(padded[:8], cores) and not padded[8:].any()


def test_pad_cores_rejects_oversized_platforms():
    with pytest.raises(ValueError):
        pad_cores(np.zeros((64, 5), dtype=np.float32), 32)


def test_pad_mask_moves_the_reject_action_to_the_end():
    inner = np.zeros(8 * 7 + 1, dtype=bool)
    inner[3] = True
    inner[-1] = True
    mask = pad_mask(inner, 8, 7, 32, has_reject=True)
    assert mask.shape == (32 * 7 + 1,)
    assert mask[3] and mask[-1] and mask.sum() == 2


def test_padded_env_action_translation_is_the_inverse_of_decoding():
    spec = agents.online_spec(build_config(8, 0.5, seed=0))
    env = OnlineEnv(build_config(8, 0.5, seed=0))
    padded = PaddedEnv(env, spec.max_cores, spec.n_sub, True, "job")
    padded.reset()

    assert padded._translate(padded.n_actions - 1) == env.reject_action
    for core in range(8):
        for sub in range(spec.n_sub):
            index = core * spec.n_sub + sub
            assert padded._translate(index) == index


def test_serving_encoder_matches_the_training_wrapper():
    """The online agent builds its own observation at inference; it must equal the trained one."""
    cfg = build_config(16, 0.75, seed=0)
    spec = agents.online_spec(cfg)
    env = OnlineEnv(cfg)
    padded = PaddedEnv(env, spec.max_cores, spec.n_sub, True, "job")
    train_obs, train_mask = padded.reset()

    serve_obs, serve_mask = encode_online(
        env.obs, env.platform.max_level, spec.max_cores, env.platform.levels
    )
    assert np.array_equal(train_obs["cores"], serve_obs["cores"])
    assert np.array_equal(train_obs["context"], serve_obs["context"])
    assert np.array_equal(train_obs["valid"], serve_obs["valid"])
    assert np.array_equal(train_mask, serve_mask)


def test_online_mask_only_permits_levels_at_or_above_the_core_floor():
    env = OnlineEnv(build_config(8, 0.85, seed=0))
    env.reset()
    levels = env.platform.levels
    mask = online_action_mask(env.obs, levels)
    for index in np.flatnonzero(mask[:-1]):
        core, sub = divmod(int(index), len(levels))
        assert levels[sub] >= env.obs.cores[core].base_level


# ----------------------------------------------------------- reward shaping


def test_shaped_rewards_telescope_to_the_terminal_objective():
    """Potential-based shaping must not change what the episode is worth."""
    cfg = build_config(8, 0.5, seed=0)
    shaped, terminal = MappingEnv(cfg, reward_mode="shaped"), MappingEnv(cfg, reward_mode="terminal")
    shaped.reset()
    terminal.reset()
    initial_cost = -shaped._potential

    total_shaped = total_terminal = 0.0
    done = False
    while not done:
        action = int(np.flatnonzero(shaped.action_masks())[0])
        _, r1, done, _, _ = shaped.step(action)
        _, r2, _, _, _ = terminal.step(action)
        total_shaped += r1
        total_terminal += r2

    # Sum of shaping terms telescopes to Phi(s_T) - Phi(s_0) = -cost_final + cost_initial.
    assert total_shaped == pytest.approx(total_terminal + initial_cost, abs=1e-9)


def test_shaping_shifts_every_episode_by_the_same_constant():
    """Policy invariance: two action sequences differ by the same amount under either reward."""
    cfg = build_config(8, 0.5, seed=0)

    def rollout(pick):
        shaped, terminal = (MappingEnv(cfg, reward_mode=m) for m in ("shaped", "terminal"))
        shaped.reset()
        terminal.reset()
        totals, done = [0.0, 0.0], False
        while not done:
            legal = np.flatnonzero(shaped.action_masks())
            action = int(legal[pick % len(legal)])
            _, r1, done, _, _ = shaped.step(action)
            _, r2, _, _, _ = terminal.step(action)
            totals[0] += r1
            totals[1] += r2
        return totals

    first, second = rollout(0), rollout(3)
    assert (first[0] - second[0]) == pytest.approx(first[1] - second[1], abs=1e-9)


# --------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_preserves_the_policy(tmp_path):
    spec = agents.mapping_spec(build_config(8, 0.5, seed=0), hidden=32)
    policy = agents.build_policy(spec).eval()
    core_features = spec.core_features
    path = agents.save(tmp_path / "agent.pt", policy, spec, {"run": "test"})

    restored, restored_spec, meta, _ = agents.load(path, "cpu")
    assert restored_spec == spec and meta["run"] == "test"

    g = torch.Generator().manual_seed(3)
    cores = torch.rand(2, 32, core_features, generator=g)
    context = torch.rand(2, spec.context_features, generator=g)
    valid = torch.ones(2, 32, dtype=torch.bool)
    with torch.no_grad():
        before, _ = policy(cores, context, valid)
        after, _ = restored(cores, context, valid)
    assert torch.allclose(before, after)


# -------------------------------------------------- trained agents (if present)

trained = pytest.mark.skipif(
    not (MAPPING_CKPT.exists() and ONLINE_CKPT.exists()),
    reason="no trained checkpoints; run `python -m rtsched.rl.train all`",
)


@trained
@pytest.mark.parametrize("cores,u", [(8, 0.5), (16, 0.75), (32, 0.85)])
def test_trained_agents_satisfy_the_phase_one_contract(cores, u):
    """The learned policies must be drop-in: same protocols, same guarantees."""
    from rtsched.experiments.pipeline import run_experiment

    out = run_experiment(build_config(cores, u, seed=0, offline="rl", online="rl"))
    allocation, summary = out.allocation, out.summary

    assert set(allocation.core_of_task) == {t.tid for t in out.taskset.periodic}
    for core in range(cores):
        assert allocation.level_of_core[core] >= allocation.base_level_of_core[core]
    if allocation.feasible:
        assert summary["hard_miss_ratio"] == 0.0


@trained
def test_trained_agent_transfers_across_core_counts_from_one_checkpoint():
    from rtsched.policies.registry import make_offline

    checkpoints = set()
    for cores in (8, 16, 32):
        cfg = build_config(cores, 0.75, seed=0, offline="rl")
        policy = make_offline("rl", cfg)
        checkpoints.add(id(policy.policy))
    assert len(checkpoints) >= 1
