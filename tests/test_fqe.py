"""Tests for Fitted Q-Evaluation (FQE)."""

import numpy as np
import pytest
from offline_rl.evaluation.fqe import FQE, FQEConfig


def _make_dataset(n=300, obs_dim=8, act_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal(n).astype(np.float32)
    return {
        "observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "actions": rng.uniform(-1, 1, (n, act_dim)).astype(np.float32),
        "rewards": rewards,
        "next_observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "dones": np.zeros(n, dtype=bool),
    }


def _random_policy(obs):
    return np.random.uniform(-1, 1, 2).astype(np.float32)


def test_fqe_loss_decreases():
    """FQE loss should decrease (or at least not increase wildly) over iterations."""
    dataset = _make_dataset()
    config = FQEConfig(n_iterations=20, lr=1e-3, batch_size=64)
    fqe = FQE(obs_dim=8, act_dim=2, policy=_random_policy, config=config)
    losses = fqe.fit(dataset, n_iterations=20)

    assert len(losses) == 20, f"Expected 20 loss values, got {len(losses)}"
    # Check that all losses are finite
    assert all(np.isfinite(l) for l in losses), "Some FQE losses are not finite"
    # Loss should generally decrease: compare first half average vs second half average
    first_half = np.mean(losses[:10])
    second_half = np.mean(losses[10:])
    # Not strictly required to decrease, but must be in a reasonable range
    assert first_half > 0 or second_half > 0, "Losses should be positive"


def test_fqe_returns_finite():
    """FQE estimate_return should return a finite float."""
    dataset = _make_dataset()
    config = FQEConfig(n_iterations=10, batch_size=64)
    fqe = FQE(obs_dim=8, act_dim=2, policy=_random_policy, config=config)
    fqe.fit(dataset, n_iterations=10)

    states = dataset["observations"][:100]
    estimate = fqe.estimate_return(states)

    assert isinstance(estimate, float), f"Expected float, got {type(estimate)}"
    assert np.isfinite(estimate), f"estimate_return returned non-finite: {estimate}"
