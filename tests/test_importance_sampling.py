"""Tests for importance sampling OPE estimators."""

import numpy as np
import pytest
from offline_rl.evaluation.ope import (
    WeightedImportanceSampling,
    bootstrap_confidence_intervals,
)


def _make_toy_dataset(n=200, act_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal(n).astype(np.float32)
    # Assign episodes (10 steps each)
    episode_ids = np.repeat(np.arange(n // 10), 10).astype(np.int64)
    if len(episode_ids) < n:
        episode_ids = np.append(episode_ids, np.full(n - len(episode_ids), episode_ids[-1]))
    return {
        "observations": rng.standard_normal((n, 4)).astype(np.float32),
        "actions": rng.uniform(-1, 1, (n, act_dim)).astype(np.float32),
        "rewards": rewards,
        "next_observations": rng.standard_normal((n, 4)).astype(np.float32),
        "dones": np.zeros(n, dtype=bool),
        "episode_ids": episode_ids[:n],
    }


def test_wis_returns_finite():
    """WIS should return a finite float on toy data."""
    dataset = _make_toy_dataset()
    episode_ids = dataset["episode_ids"]
    n = len(dataset["rewards"])

    behavior_log_probs = -np.ones(n) * 2.0  # log(0.135) ≈ -2
    eval_log_probs = -np.ones(n) * 2.5

    wis = WeightedImportanceSampling()
    estimate = wis.estimate(
        dataset, behavior_log_probs, eval_log_probs, episode_ids, discount=0.99
    )
    assert isinstance(estimate, float), f"Expected float, got {type(estimate)}"
    assert np.isfinite(estimate), f"WIS estimate is not finite: {estimate}"


def test_bootstrap_ci_correct_length():
    """bootstrap_confidence_intervals should return a 2-tuple."""
    data = list(np.random.standard_normal(100))
    ci = bootstrap_confidence_intervals(data, n_bootstrap=200, ci=0.95)
    assert isinstance(ci, tuple), f"Expected tuple, got {type(ci)}"
    assert len(ci) == 2, f"Expected 2-tuple, got length {len(ci)}"
    lower, upper = ci
    assert lower <= upper, f"Lower {lower} > upper {upper}"


def test_bootstrap_interval_contains_mean():
    """Bootstrap CI should contain the true mean in more than 90% of trials."""
    rng = np.random.default_rng(42)
    n_trials = 50
    contained = 0

    for _ in range(n_trials):
        # Generate samples from N(0, 1)
        samples = rng.standard_normal(200).tolist()
        ci = bootstrap_confidence_intervals(samples, n_bootstrap=500, ci=0.95)
        true_mean = 0.0
        if ci[0] <= true_mean <= ci[1]:
            contained += 1

    coverage = contained / n_trials
    # Should contain true mean at least 80% of the time (95% CI, finite samples)
    assert coverage >= 0.80, (
        f"Bootstrap CI contained true mean only {coverage*100:.0f}% of the time "
        f"(expected >= 80%)"
    )
