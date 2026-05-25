"""Tests for Decision Transformer model and algorithm."""

import numpy as np
import pytest
import torch

from offline_rl.models.transformer import DecisionTransformerModel
from offline_rl.algorithms.decision_transformer import DecisionTransformer, DTConfig
from offline_rl.datasets.trajectory_buffer import TrajectoryReplayBuffer


# ── Fixtures ─────────────────────────────────────────────────────────────────

OBS_DIM = 8
ACT_DIM = 3
BATCH = 4
CONTEXT = 5


def make_model():
    return DecisionTransformerModel(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        d_model=32,
        n_layers=1,
        n_heads=1,
        context_len=CONTEXT,
        max_ep_len=100,
        dropout=0.0,
    )


def make_batch(B=BATCH, T=CONTEXT):
    states = torch.randn(B, T, OBS_DIM)
    actions = torch.randn(B, T, ACT_DIM)
    rtgs = torch.randn(B, T, 1)
    timesteps = torch.randint(0, 100, (B, T))
    return states, actions, rtgs, timesteps


def make_trajectory_dataset(n_episodes=10, ep_len=20):
    """Build a minimal dataset with episode_ids."""
    obs_list, act_list, rew_list, done_list, ep_id_list = [], [], [], [], []
    for ep in range(n_episodes):
        obs_list.append(np.random.randn(ep_len, OBS_DIM).astype(np.float32))
        act_list.append(np.random.randn(ep_len, ACT_DIM).astype(np.float32))
        rew_list.append(np.random.randn(ep_len).astype(np.float32))
        done_list.append(np.zeros(ep_len, dtype=np.float32))
        ep_id_list.append(np.full(ep_len, ep, dtype=np.int64))
    return {
        "observations": np.concatenate(obs_list),
        "actions": np.concatenate(act_list),
        "rewards": np.concatenate(rew_list),
        "dones": np.concatenate(done_list),
        "episode_ids": np.concatenate(ep_id_list),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_forward_pass_shapes():
    """model(states, actions, rtgs, timesteps) returns (B, T, act_dim)."""
    model = make_model()
    states, actions, rtgs, timesteps = make_batch()

    out = model(states, actions, rtgs, timesteps)

    assert out.shape == (BATCH, CONTEXT, ACT_DIM), (
        f"Expected shape ({BATCH}, {CONTEXT}, {ACT_DIM}), got {out.shape}"
    )


def test_action_output_bounded():
    """With action_tanh=True outputs should be in [-1, 1]."""
    model = make_model()
    states, actions, rtgs, timesteps = make_batch()
    out = model(states, actions, rtgs, timesteps)
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <= 1.0 + 1e-5


def test_training_step_loss_finite():
    """train_step on a random batch returns a finite scalar loss."""
    cfg = DTConfig(
        context_len=CONTEXT,
        d_model=32,
        n_layers=1,
        n_heads=1,
        lr=1e-4,
        batch_size=BATCH,
        max_ep_len=100,
    )
    agent = DecisionTransformer(OBS_DIM, ACT_DIM, cfg, device="cpu")

    states, actions, rtgs, timesteps = make_batch()
    batch = {
        "observations": states,
        "actions": actions,
        "returns_to_go": rtgs,
        "timesteps": timesteps,
    }
    metrics = agent.train_step(batch)

    assert "action_loss" in metrics
    loss = metrics["action_loss"]
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss >= 0.0


def test_training_step_loss_decreases():
    """Multiple train steps should reduce the loss (at least not diverge)."""
    cfg = DTConfig(
        context_len=CONTEXT,
        d_model=32,
        n_layers=1,
        n_heads=1,
        lr=1e-3,
        batch_size=BATCH,
        max_ep_len=100,
    )
    agent = DecisionTransformer(OBS_DIM, ACT_DIM, cfg, device="cpu")
    states, actions, rtgs, timesteps = make_batch()
    batch = {
        "observations": states,
        "actions": actions,
        "returns_to_go": rtgs,
        "timesteps": timesteps,
    }

    first_loss = agent.train_step(batch)["action_loss"]
    for _ in range(19):
        last_loss = agent.train_step(batch)["action_loss"]

    # Training on same batch: final loss should be ≤ 2× first loss (not diverged)
    assert last_loss <= first_loss * 2, (
        f"Loss diverged from {first_loss:.4f} to {last_loss:.4f}"
    )


def test_different_target_returns_produce_different_actions():
    """High RTG vs low RTG should produce different select_action outputs."""
    cfg = DTConfig(
        context_len=CONTEXT,
        d_model=32,
        n_layers=1,
        n_heads=1,
        max_ep_len=100,
    )
    agent = DecisionTransformer(OBS_DIM, ACT_DIM, cfg, device="cpu")

    # Build history arrays
    T = CONTEXT
    obs_history = np.random.randn(T, OBS_DIM).astype(np.float32)
    act_history = np.random.randn(T, ACT_DIM).astype(np.float32)

    high_rtg = np.full(T, 200.0, dtype=np.float32)
    low_rtg = np.full(T, -200.0, dtype=np.float32)

    action_high = agent.select_action(obs_history, act_history, high_rtg, timestep=T - 1)
    action_low = agent.select_action(obs_history, act_history, low_rtg, timestep=T - 1)

    # Actions must differ (different conditioning)
    assert not np.allclose(action_high, action_low, atol=1e-5), (
        "High RTG and low RTG produced identical actions — conditioning not working"
    )
    assert action_high.shape == (ACT_DIM,)
    assert action_low.shape == (ACT_DIM,)


def test_trajectory_replay_buffer_sample_shapes():
    """TrajectoryReplayBuffer.sample returns correctly shaped tensors."""
    buf = TrajectoryReplayBuffer(context_len=CONTEXT, device="cpu")
    dataset = make_trajectory_dataset(n_episodes=5, ep_len=30)
    buf.load_from_dataset(dataset)

    batch = buf.sample(BATCH)

    assert batch["observations"].shape == (BATCH, CONTEXT, OBS_DIM)
    assert batch["actions"].shape == (BATCH, CONTEXT, ACT_DIM)
    assert batch["returns_to_go"].shape == (BATCH, CONTEXT, 1)
    assert batch["timesteps"].shape == (BATCH, CONTEXT)


def test_trajectory_buffer_short_episode_padding():
    """Episodes shorter than context_len should be zero-padded."""
    buf = TrajectoryReplayBuffer(context_len=10, device="cpu")
    # Only 3-step episodes
    dataset = {
        "observations": np.random.randn(3, OBS_DIM).astype(np.float32),
        "actions": np.random.randn(3, ACT_DIM).astype(np.float32),
        "rewards": np.random.randn(3).astype(np.float32),
        "dones": np.zeros(3, dtype=np.float32),
        "episode_ids": np.zeros(3, dtype=np.int64),
    }
    buf.load_from_dataset(dataset)
    batch = buf.sample(2)

    assert batch["observations"].shape == (2, 10, OBS_DIM)
    # First 7 positions should be zeros (padding)
    pad = batch["observations"][0, :7, :]
    assert torch.all(pad == 0.0), "Padding should be zeros"


def test_dt_save_load(tmp_path):
    """save/load round-trip preserves model state."""
    cfg = DTConfig(context_len=CONTEXT, d_model=32, n_layers=1, n_heads=1, max_ep_len=100)
    agent = DecisionTransformer(OBS_DIM, ACT_DIM, cfg)

    ckpt = str(tmp_path / "dt.pt")
    agent.save(ckpt)

    loaded = DecisionTransformer.load(ckpt)
    assert loaded.obs_dim == OBS_DIM
    assert loaded.act_dim == ACT_DIM

    # Predictions should match
    obs_h = np.random.randn(CONTEXT, OBS_DIM).astype(np.float32)
    act_h = np.random.randn(CONTEXT, ACT_DIM).astype(np.float32)
    rtg_h = np.ones(CONTEXT, dtype=np.float32)

    agent.model.eval()
    loaded.model.eval()
    a1 = agent.select_action(obs_h, act_h, rtg_h, CONTEXT - 1)
    a2 = loaded.select_action(obs_h, act_h, rtg_h, CONTEXT - 1)
    np.testing.assert_allclose(a1, a2, atol=1e-5)
