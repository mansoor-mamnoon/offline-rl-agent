import os
import tempfile
import numpy as np
import torch
import pytest

from offline_rl.datasets.replay_buffer import ReplayBuffer
from offline_rl.datasets.normalization import (
    normalize_dataset,
    denormalize_rewards,
    save_normalization_stats,
    load_normalization_stats,
)


def make_dummy_dataset(n=200, obs_dim=32, act_dim=4):
    return {
        "observations": np.random.randn(n, obs_dim).astype(np.float32),
        "actions": np.random.randn(n, act_dim).astype(np.float32),
        "rewards": np.random.randn(n).astype(np.float32),
        "next_observations": np.random.randn(n, obs_dim).astype(np.float32),
        "dones": np.zeros(n, dtype=bool),
        "episode_ids": np.arange(n, dtype=np.int64),
    }


def test_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=100, obs_dim=4, act_dim=2)
    for _ in range(50):
        obs = np.random.randn(4).astype(np.float32)
        act = np.random.randn(2).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        buf.add(obs, act, 1.0, next_obs, False)

    assert len(buf) == 50
    batch = buf.sample(32)
    assert batch["observations"].shape == (32, 4)
    assert batch["actions"].shape == (32, 2)
    assert batch["rewards"].shape == (32,)
    assert batch["next_observations"].shape == (32, 4)
    assert batch["dones"].shape == (32,)


def test_sample_returns_tensors():
    buf = ReplayBuffer(capacity=100, obs_dim=4, act_dim=2, device="cpu")
    for _ in range(20):
        buf.add(
            np.zeros(4), np.zeros(2), 0.0, np.zeros(4), False
        )
    batch = buf.sample(10)
    for v in batch.values():
        assert isinstance(v, torch.Tensor)


def test_load_from_dataset():
    dataset = make_dummy_dataset(n=200, obs_dim=32, act_dim=4)
    buf = ReplayBuffer(capacity=500, obs_dim=32, act_dim=4)
    buf.load_from_dataset(dataset)
    assert len(buf) == 200
    batch = buf.sample(64)
    assert batch["observations"].shape == (64, 32)


def test_normalization_roundtrip():
    dataset = make_dummy_dataset(n=100, obs_dim=8, act_dim=4)
    norm_dataset, stats = normalize_dataset(dataset)

    # Normalized observations should be approximately zero-mean
    assert abs(norm_dataset["observations"].mean()) < 0.5

    # Denormalize rewards
    recovered = denormalize_rewards(norm_dataset["rewards"], stats)
    np.testing.assert_allclose(recovered, dataset["rewards"], atol=1e-4)


def test_normalization_stats_save_load():
    dataset = make_dummy_dataset(n=50, obs_dim=4, act_dim=2)
    _, stats = normalize_dataset(dataset)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "stats.json")
        save_normalization_stats(stats, path)
        loaded = load_normalization_stats(path)
    np.testing.assert_allclose(stats["obs_mean"], loaded["obs_mean"], atol=1e-6)
    assert abs(stats["reward_mean"] - loaded["reward_mean"]) < 1e-6


def test_capacity_overflow_handling():
    buf = ReplayBuffer(capacity=10, obs_dim=2, act_dim=1)
    for i in range(25):
        buf.add(
            np.array([i, i], dtype=np.float32),
            np.array([i], dtype=np.float32),
            float(i),
            np.array([i, i], dtype=np.float32),
            False,
        )
    # Buffer should hold at most capacity items
    assert len(buf) == 10
    batch = buf.sample(10)
    assert batch["observations"].shape == (10, 2)


def test_load_discrete_actions():
    """Discrete actions (1D integer array) should be loaded as 2D float."""
    n = 50
    dataset = {
        "observations": np.random.randn(n, 4).astype(np.float32),
        "actions": np.random.randint(0, 4, n).astype(np.int64),  # discrete
        "rewards": np.random.randn(n).astype(np.float32),
        "next_observations": np.random.randn(n, 4).astype(np.float32),
        "dones": np.zeros(n, dtype=bool),
    }
    buf = ReplayBuffer(capacity=100, obs_dim=4, act_dim=1)
    buf.load_from_dataset(dataset)
    assert len(buf) == n
    batch = buf.sample(10)
    assert batch["actions"].shape == (10, 1)
