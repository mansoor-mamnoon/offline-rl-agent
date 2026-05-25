"""Tests for dataset diagnostics module."""

import json
import numpy as np
import pytest

from offline_rl.datasets.diagnostics import DatasetDiagnostics, DiagnosticsReport


def _make_dataset(n=500, obs_dim=8, act_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal(n).astype(np.float32)
    dones = (rng.random(n) < 0.05).astype(bool)
    episode_ids = np.cumsum(dones).astype(np.int64)
    return {
        "observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "actions": rng.uniform(-1, 1, (n, act_dim)).astype(np.float32),
        "rewards": rewards,
        "next_observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "dones": dones,
        "episode_ids": episode_ids,
    }


def test_coverage_score_in_range():
    """Coverage score must be a float in [0, 1]."""
    dataset = _make_dataset()
    diag = DatasetDiagnostics(dataset)
    score = diag.compute_coverage_score()
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Coverage score {score} not in [0, 1]"


def test_behavior_entropy_positive():
    """Behavior entropy must be a positive float."""
    dataset = _make_dataset()
    diag = DatasetDiagnostics(dataset)
    entropy = diag.compute_behavior_entropy()
    assert isinstance(entropy, float), f"Expected float, got {type(entropy)}"
    assert entropy >= 0.0, f"Entropy {entropy} is negative"


def test_outlier_detection_finds_injected():
    """Injecting extreme reward outliers should be detected."""
    rng = np.random.default_rng(42)
    n = 1000
    rewards = rng.standard_normal(n).astype(np.float32)
    # Inject clear outliers
    outlier_indices_injected = [10, 50, 200, 700, 999]
    for idx in outlier_indices_injected:
        rewards[idx] = 1000.0  # extreme positive outlier

    dataset = {
        "observations": rng.standard_normal((n, 4)).astype(np.float32),
        "actions": rng.uniform(-1, 1, (n, 2)).astype(np.float32),
        "rewards": rewards,
        "next_observations": rng.standard_normal((n, 4)).astype(np.float32),
        "dones": np.zeros(n, dtype=bool),
        "episode_ids": np.zeros(n, dtype=np.int64),
    }

    diag = DatasetDiagnostics(dataset)
    detected = diag.detect_outlier_transitions()

    assert len(detected) > 0, "Should detect at least some outliers"
    # At least one injected outlier should be found
    detected_set = set(detected)
    injected_set = set(outlier_indices_injected)
    overlap = detected_set & injected_set
    assert len(overlap) > 0, (
        f"None of the injected outlier indices {injected_set} found in detected {detected_set}"
    )


def test_report_serializes():
    """DiagnosticsReport.to_dict() must produce valid JSON-serializable output."""
    dataset = _make_dataset()
    diag = DatasetDiagnostics(dataset)
    report = diag.run_all()

    d = report.to_dict()
    # Must be JSON serializable
    serialized = json.dumps(d)
    loaded = json.loads(serialized)

    assert "n_transitions" in loaded
    assert "coverage_score" in loaded
    assert "ood_risk" in loaded
    assert "reward_stats" in loaded
    assert isinstance(loaded["state_pca_points"], list)
