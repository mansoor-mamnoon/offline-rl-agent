"""Safety evaluation metrics for offline RL policies."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SafetyReport:
    mean_return: float
    std_return: float
    worst_5pct_return: float     # CVaR at 5%
    slo_violation_rate: float
    ood_action_rate: float
    catastrophic_failure_rate: float
    episode_returns: np.ndarray

    def to_dict(self) -> dict:
        d = {
            k: v
            for k, v in self.__dict__.items()
            if not isinstance(v, np.ndarray)
        }
        d["episode_returns"] = self.episode_returns.tolist()
        return d

    def summary_string(self) -> str:
        return (
            f"Safety Report\n"
            f"  Mean return:       {self.mean_return:.3f} ± {self.std_return:.3f}\n"
            f"  CVaR 5%:           {self.worst_5pct_return:.3f}\n"
            f"  SLO violation:     {self.slo_violation_rate*100:.1f}%\n"
            f"  OOD action rate:   {self.ood_action_rate*100:.1f}%\n"
            f"  Catastrophic fail: {self.catastrophic_failure_rate*100:.1f}%\n"
        )


class SafetyEvaluator:
    """Evaluates a policy using safety-oriented metrics."""

    def __init__(
        self,
        env,
        policy,
        support_estimator=None,
        ood_threshold: float = 0.3,
    ):
        self.env = env
        self.policy = policy
        self.support_estimator = support_estimator
        self.ood_threshold = ood_threshold

    def evaluate(
        self,
        n_episodes: int = 50,
        failure_threshold: float = -50.0,
    ) -> SafetyReport:
        """Run n_episodes and collect safety metrics.

        Args:
            n_episodes: Number of evaluation episodes.
            failure_threshold: Episodes with return below this are catastrophic.

        Returns:
            SafetyReport with aggregated metrics.
        """
        episode_returns = []
        ood_counts = []
        slo_violations = []

        for ep in range(n_episodes):
            obs = self.env.reset()
            total_reward = 0.0
            ep_ood = 0
            ep_steps = 0
            slo_viol = 0
            done = False

            while not done:
                action = self.policy.select_action(obs)

                # Check support
                if (
                    self.support_estimator is not None
                    and self.support_estimator._fitted
                ):
                    score = self.support_estimator.support_score(obs, action)
                    if score < self.ood_threshold:
                        ep_ood += 1

                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                slo_viol += int(info.get("slo_violated", False))
                ep_steps += 1
                obs = next_obs

            episode_returns.append(total_reward)
            ood_counts.append(ep_ood / max(ep_steps, 1))
            slo_violations.append(slo_viol / max(ep_steps, 1))

        returns = np.array(episode_returns)
        n_fail = max(1, int(len(returns) * 0.05))
        worst_5pct = float(np.sort(returns)[:n_fail].mean())

        return SafetyReport(
            mean_return=float(returns.mean()),
            std_return=float(returns.std()),
            worst_5pct_return=worst_5pct,
            slo_violation_rate=float(np.mean(slo_violations)),
            ood_action_rate=float(np.mean(ood_counts)),
            catastrophic_failure_rate=float((returns < failure_threshold).mean()),
            episode_returns=returns,
        )
