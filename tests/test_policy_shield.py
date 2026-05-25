"""Tests for PolicyShield safety wrapper."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from offline_rl.safety.policy_shield import PolicyShield, ShieldedPolicy


# ── Helpers ────────────────────────────────────────────────────────────────

OBS_DIM = 8
ACT_DIM = 4


def make_obs():
    return np.random.randn(OBS_DIM).astype(np.float32)


def make_action():
    return np.random.randn(ACT_DIM).astype(np.float32)


def make_support_estimator(score: float):
    """Return a mock support estimator with a fixed score."""
    est = MagicMock()
    est.support_score.return_value = score
    est.batch_support_scores.return_value = np.full(51, score)  # n_candidates + maybe beh
    est._fitted = True
    return est


def make_behavior_policy():
    """Return a callable that always returns a fixed action."""
    fixed = np.ones(ACT_DIM, dtype=np.float32) * 0.5

    def policy(obs):
        return fixed.copy()

    return policy


# ── Tests ─────────────────────────────────────────────────────────────────


def test_intervention_triggers_when_low_support():
    """Mock support_estimator returning 0.0: shield should intervene."""
    est = make_support_estimator(0.0)
    shield = PolicyShield(support_estimator=est, threshold=0.3)

    obs = make_obs()
    action = make_action()
    _, was_intervened = shield.shield(obs, action)

    assert was_intervened is True, "Expected intervention when support score is 0.0"


def test_no_intervention_when_in_distribution():
    """Mock score=0.9: shield should NOT intervene."""
    est = make_support_estimator(0.9)
    shield = PolicyShield(support_estimator=est, threshold=0.3)

    obs = make_obs()
    action = make_action()
    returned_action, was_intervened = shield.shield(obs, action)

    assert was_intervened is False, "Expected no intervention when support score is 0.9"
    np.testing.assert_array_equal(returned_action, action)


def test_intervention_rate_tracking():
    """Run 10 steps, check get_stats() counts are consistent."""
    # 5 below threshold, 5 above
    call_count = [0]

    def score_fn(*args, **kwargs):
        c = call_count[0]
        call_count[0] += 1
        return 0.0 if c < 5 else 0.9

    est = MagicMock()
    est.support_score.side_effect = score_fn
    est.batch_support_scores.return_value = np.zeros(50)
    est._fitted = True

    shield = PolicyShield(support_estimator=est, threshold=0.3)

    obs = make_obs()
    action = make_action()
    for _ in range(10):
        shield.shield(obs, action)

    stats = shield.get_stats()
    assert stats["total_steps"] == 10
    assert stats["interventions"] == 5
    assert abs(stats["intervention_rate"] - 0.5) < 1e-6


def test_reset_stats():
    """reset_stats() zeroes out counters."""
    est = make_support_estimator(0.0)
    shield = PolicyShield(support_estimator=est, threshold=0.3)
    obs = make_obs()
    action = make_action()
    for _ in range(5):
        shield.shield(obs, action)

    shield.reset_stats()
    stats = shield.get_stats()
    assert stats["total_steps"] == 0
    assert stats["interventions"] == 0
    assert stats["intervention_rate"] == 0.0


def test_all_strategies_return_valid_actions():
    """nearest_safe, blend, reject all return an action with the same shape as input."""
    obs = make_obs()
    action = make_action()
    beh_policy = make_behavior_policy()

    for strategy in ["nearest_safe", "blend", "reject"]:
        est = make_support_estimator(0.0)
        # batch_support_scores: enough elements for n_candidates (+1 for beh)
        est.batch_support_scores.return_value = np.zeros(51)
        shield = PolicyShield(
            support_estimator=est,
            behavior_policy=beh_policy,
            threshold=0.3,
            intervention_strategy=strategy,
            n_candidates=50,
        )
        returned_action, was_intervened = shield.shield(obs, action)
        assert was_intervened is True, f"Expected intervention for strategy={strategy}"
        assert returned_action.shape == action.shape, (
            f"Strategy {strategy} returned shape {returned_action.shape}, expected {action.shape}"
        )


def test_nearest_safe_without_behavior_policy():
    """nearest_safe still works when behavior_policy is None."""
    est = make_support_estimator(0.0)
    est.batch_support_scores.return_value = np.zeros(50)
    shield = PolicyShield(
        support_estimator=est,
        behavior_policy=None,
        threshold=0.3,
        intervention_strategy="nearest_safe",
    )
    obs = make_obs()
    action = make_action()
    returned_action, _ = shield.shield(obs, action)
    assert returned_action.shape == action.shape


def test_blend_without_behavior_policy():
    """blend falls back gracefully when behavior_policy is None."""
    est = make_support_estimator(0.1)  # below threshold 0.3
    shield = PolicyShield(
        support_estimator=est,
        behavior_policy=None,
        threshold=0.3,
        intervention_strategy="blend",
    )
    obs = make_obs()
    action = make_action()
    returned_action, was_intervened = shield.shield(obs, action)
    assert was_intervened is True
    assert returned_action.shape == action.shape


def test_reject_without_behavior_policy():
    """reject falls back gracefully when behavior_policy is None."""
    est = make_support_estimator(0.0)
    shield = PolicyShield(
        support_estimator=est,
        behavior_policy=None,
        threshold=0.3,
        intervention_strategy="reject",
    )
    obs = make_obs()
    action = make_action()
    returned_action, _ = shield.shield(obs, action)
    assert returned_action.shape == action.shape


def test_wrap_creates_shielded_policy():
    """wrap() returns a ShieldedPolicy that calls shield on each select_action."""
    est = make_support_estimator(0.0)
    est.batch_support_scores.return_value = np.zeros(50)
    shield = PolicyShield(support_estimator=est, threshold=0.3)

    mock_policy = MagicMock()
    mock_policy.select_action.return_value = make_action()

    shielded = shield.wrap(mock_policy)
    assert isinstance(shielded, ShieldedPolicy)

    obs = make_obs()
    shielded.select_action(obs)

    mock_policy.select_action.assert_called_once()
    assert shield.get_stats()["interventions"] == 1


def test_intervention_rate_is_zero_when_all_in_support():
    """Intervention rate should be 0 when all actions are in support."""
    est = make_support_estimator(1.0)
    shield = PolicyShield(support_estimator=est, threshold=0.3)
    obs = make_obs()
    action = make_action()
    for _ in range(10):
        shield.shield(obs, action)
    assert shield.get_stats()["intervention_rate"] == 0.0
