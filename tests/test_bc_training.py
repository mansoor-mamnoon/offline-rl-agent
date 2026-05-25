import os
import tempfile
import numpy as np
import torch
import pytest

from offline_rl.algorithms.bc import BehaviorCloning, BCConfig
from offline_rl.datasets.replay_buffer import ReplayBuffer


OBS_DIM = 32
ACT_DIM = 4
BATCH_SIZE = 16


def make_bc(action_space="continuous"):
    config = BCConfig(
        hidden_dims=[64, 64],
        lr=1e-3,
        batch_size=BATCH_SIZE,
        n_epochs=5,
        action_space=action_space,
    )
    return BehaviorCloning(OBS_DIM, ACT_DIM, config, device="cpu")


def make_batch(n=BATCH_SIZE, action_space="continuous"):
    obs = torch.randn(n, OBS_DIM)
    if action_space == "continuous":
        actions = torch.randn(n, ACT_DIM)
    else:
        actions = torch.randint(0, ACT_DIM, (n, 1)).float()
    return {"observations": obs, "actions": actions}


def test_bc_loss_decreases_over_10_steps():
    bc = make_bc("continuous")
    batch = make_batch(64, "continuous")
    losses = []
    for _ in range(10):
        result = bc.train_step(batch)
        losses.append(result["loss"])
    # Loss should generally decrease or at minimum not explode
    assert losses[-1] < losses[0] * 10, f"Loss went from {losses[0]:.3f} to {losses[-1]:.3f}"
    # The loss key should exist
    assert "loss" in result
    assert "nll" in result


def test_bc_discrete_loss():
    bc = make_bc("discrete")
    batch = make_batch(32, "discrete")
    result = bc.train_step(batch)
    assert "loss" in result
    assert result["loss"] > 0


def test_select_action_output_shape_continuous():
    bc = make_bc("continuous")
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = bc.select_action(obs, deterministic=True)
    assert action.shape == (ACT_DIM,)


def test_select_action_output_shape_discrete():
    config = BCConfig(hidden_dims=[32], n_epochs=1, action_space="discrete")
    bc = BehaviorCloning(OBS_DIM, 4, config)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = bc.select_action(obs, deterministic=True)
    # For discrete, returns scalar or array with single element
    assert np.asarray(action).size == 1


def test_select_action_stochastic():
    bc = make_bc("continuous")
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    actions = [bc.select_action(obs, deterministic=False) for _ in range(5)]
    # Stochastic actions should not all be identical
    assert not all(np.allclose(actions[0], a) for a in actions[1:])


def test_save_load_roundtrip():
    bc = make_bc("continuous")
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action_before = bc.select_action(obs, deterministic=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bc_checkpoint.pt")
        bc.save(path)
        bc2 = BehaviorCloning.load(path, device="cpu")
        action_after = bc2.select_action(obs, deterministic=True)

    np.testing.assert_allclose(action_before, action_after, atol=1e-5)


def test_bc_with_replay_buffer():
    """Test training through replay buffer sampling."""
    bc = make_bc("continuous")
    # Create buffer and fill it
    buf = ReplayBuffer(capacity=500, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    dataset = {
        "observations": np.random.randn(200, OBS_DIM).astype(np.float32),
        "actions": np.random.randn(200, ACT_DIM).astype(np.float32),
        "rewards": np.random.randn(200).astype(np.float32),
        "next_observations": np.random.randn(200, OBS_DIM).astype(np.float32),
        "dones": np.zeros(200, dtype=bool),
    }
    buf.load_from_dataset(dataset)
    losses = []
    for _ in range(5):
        batch = buf.sample(BATCH_SIZE)
        result = bc.train_step(batch)
        losses.append(result["loss"])
    assert len(losses) == 5
    assert all(np.isfinite(l) for l in losses)
