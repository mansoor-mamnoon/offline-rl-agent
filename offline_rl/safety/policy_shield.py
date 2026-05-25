import numpy as np
from typing import Callable, Optional, Tuple


class PolicyShield:
    def __init__(
        self,
        support_estimator,
        behavior_policy: Optional[Callable] = None,
        threshold: float = 0.3,
        intervention_strategy: str = "nearest_safe",
        n_candidates: int = 50,
    ):
        self.support_estimator = support_estimator
        self.behavior_policy = behavior_policy
        self.threshold = threshold
        self.strategy = intervention_strategy
        self.n_candidates = n_candidates
        self._total_steps = 0
        self._interventions = 0

    def shield(self, obs: np.ndarray, proposed_action: np.ndarray) -> Tuple[np.ndarray, bool]:
        score = self.support_estimator.support_score(obs, proposed_action)
        self._total_steps += 1

        if score >= self.threshold:
            return proposed_action, False

        self._interventions += 1

        if self.strategy == "nearest_safe":
            return self._nearest_safe(obs, proposed_action), True
        elif self.strategy == "blend":
            return self._blend(obs, proposed_action, score), True
        elif self.strategy == "reject":
            return self._reject(obs, proposed_action), True
        else:
            return proposed_action, False

    def _nearest_safe(self, obs, proposed_action):
        # Sample candidates from neighborhood of proposed action + behavior policy
        act_dim = proposed_action.shape[0]
        candidates = []

        # Random candidates around proposed action
        noise = np.random.randn(self.n_candidates, act_dim) * 0.2
        rand_candidates = np.clip(proposed_action + noise, -1, 1)
        candidates.extend(rand_candidates)

        # Behavior policy candidate
        if self.behavior_policy is not None:
            beh_action = self.behavior_policy(obs)
            candidates.append(beh_action)

        # Find highest support score
        scores = self.support_estimator.batch_support_scores(
            np.tile(obs, (len(candidates), 1)),
            np.array(candidates),
        )
        best_idx = np.argmax(scores)
        best_action = candidates[best_idx]
        return best_action

    def _blend(self, obs, proposed_action, score):
        alpha = max(0, 1 - score / self.threshold)
        if self.behavior_policy is not None:
            beh_action = self.behavior_policy(obs)
            return (1 - alpha) * proposed_action + alpha * beh_action
        return proposed_action

    def _reject(self, obs, proposed_action):
        if self.behavior_policy is not None:
            return self.behavior_policy(obs)
        return proposed_action

    def get_stats(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "interventions": self._interventions,
            "intervention_rate": self._interventions / max(self._total_steps, 1),
        }

    def reset_stats(self):
        self._total_steps = 0
        self._interventions = 0

    def wrap(self, policy) -> "ShieldedPolicy":
        return ShieldedPolicy(policy, self)


class ShieldedPolicy:
    def __init__(self, policy, shield: PolicyShield):
        self.policy = policy
        self.shield = shield

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        proposed = self.policy.select_action(obs, deterministic)
        safe_action, _ = self.shield.shield(obs, proposed)
        return safe_action
