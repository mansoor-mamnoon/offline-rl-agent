from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    support_scores: Optional[np.ndarray] = None
    constraint_violations: Optional[np.ndarray] = None  # (T, n_constraints)
    total_return: float = 0.0
    is_failure: bool = False


@dataclass
class FailureExplanation:
    failure_step: int
    cause: str  # 'ood_action' | 'constraint_violation' | 'incident_unhandled' | 'cascading_overload' | 'unknown'
    evidence: dict
    counterfactual: str


class FailureExplorer:
    def __init__(
        self,
        policy,
        env,
        support_estimator=None,
        constraint_critic=None,
        failure_threshold: float = -50.0,
    ):
        self.policy = policy
        self.env = env
        self.support_estimator = support_estimator
        self.constraint_critic = constraint_critic
        self.failure_threshold = failure_threshold
        self.episodes: list = []

    def collect_episodes(self, n_episodes: int = 100) -> list:
        self.episodes = []

        for _ in range(n_episodes):
            obs = self.env.reset()
            states, actions, rewards, dones = [], [], [], []
            support_scores, constraint_violations = [], []
            done = False

            while not done:
                action = self.policy.select_action(obs)

                if self.support_estimator is not None and self.support_estimator._fitted:
                    score = self.support_estimator.support_score(obs, action)
                    support_scores.append(score)

                if self.constraint_critic is not None:
                    violations = self.constraint_critic.estimate_constraint_violation(obs, action)
                    constraint_violations.append(violations)

                next_obs, reward, done, _ = self.env.step(action)
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(float(done))
                obs = next_obs

            total_return = sum(rewards)
            ep = Episode(
                states=np.array(states),
                actions=np.array(actions),
                rewards=np.array(rewards),
                dones=np.array(dones),
                support_scores=np.array(support_scores) if support_scores else None,
                constraint_violations=(
                    np.array(constraint_violations) if constraint_violations else None
                ),
                total_return=total_return,
                is_failure=total_return < self.failure_threshold,
            )
            self.episodes.append(ep)

        return self.episodes

    def get_failure_episodes(self, k: int = 10) -> list:
        return sorted(self.episodes, key=lambda e: e.total_return)[:k]

    def explain_failure(self, episode: Episode) -> FailureExplanation:
        rewards = episode.rewards

        # Find failure step: where cumulative reward starts consistently negative
        cumulative = np.cumsum(rewards)
        if len(cumulative) > 0:
            failure_step = int(np.argmin(cumulative))
        else:
            failure_step = 0

        # Determine cause
        cause = "unknown"
        evidence = {}

        if episode.support_scores is not None and len(episode.support_scores) > 0:
            low_support_steps = np.where(episode.support_scores < 0.3)[0]
            if len(low_support_steps) > 0 and low_support_steps[0] <= failure_step:
                cause = "ood_action"
                evidence["low_support_steps"] = low_support_steps.tolist()[:5]
                evidence["min_support_score"] = float(episode.support_scores.min())

        if cause == "unknown" and episode.constraint_violations is not None:
            violations = episode.constraint_violations
            high_violations = np.where(violations.max(axis=1) > 0.5)[0]
            if len(high_violations) > 0 and high_violations[0] <= failure_step:
                cause = "constraint_violation"
                evidence["violated_constraints"] = high_violations.tolist()[:5]

        if cause == "unknown":
            # Check for cascading (load increases over time in states)
            if episode.states.shape[1] >= 3:
                loads = episode.states[:, :3]  # first 3 dims are loads in traffic env
                if loads[-1].mean() > loads[0].mean() + 0.3:
                    cause = "cascading_overload"
                    evidence["load_increase"] = float(loads[-1].mean() - loads[0].mean())

        counterfactual = {
            "ood_action": "Use behavior policy actions when support score falls below threshold",
            "constraint_violation": "Apply constraint critic to filter actions before execution",
            "cascading_overload": "Shed traffic earlier when load exceeds 0.7 on any backend",
            "incident_unhandled": "Monitor incident_flag and reroute traffic immediately",
            "unknown": "Investigate state trajectory for anomalies",
        }[cause]

        return FailureExplanation(
            failure_step=failure_step,
            cause=cause,
            evidence=evidence,
            counterfactual=counterfactual,
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, ep in enumerate(self.episodes):
            row = {
                "episode_id": i,
                "total_return": ep.total_return,
                "is_failure": ep.is_failure,
                "n_steps": len(ep.rewards),
                "mean_reward": ep.rewards.mean(),
            }
            if ep.support_scores is not None:
                row["mean_support_score"] = ep.support_scores.mean()
                row["min_support_score"] = ep.support_scores.min()
            rows.append(row)
        return pd.DataFrame(rows)
