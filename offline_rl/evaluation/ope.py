"""Offline Policy Evaluation (OPE) with WIS, FQE, and Doubly Robust estimators."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class OPEResult:
    policy_name: str
    fqe_estimate: Optional[float]
    wis_estimate: Optional[float]
    dr_estimate: Optional[float]
    bootstrap_ci_95: tuple
    unsafe_action_rate: float
    ood_action_fraction: float

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "fqe_estimate": self.fqe_estimate,
            "wis_estimate": self.wis_estimate,
            "dr_estimate": self.dr_estimate,
            "bootstrap_ci_95": list(self.bootstrap_ci_95),
            "unsafe_action_rate": self.unsafe_action_rate,
            "ood_action_fraction": self.ood_action_fraction,
        }


class ImportanceSampling:
    """Per-episode importance sampling estimator (IS)."""

    def estimate(
        self,
        dataset: dict,
        behavior_log_probs: np.ndarray,
        eval_log_probs: np.ndarray,
        episode_ids: np.ndarray,
        discount: float = 0.99,
        max_ratio: float = 20.0,
    ) -> float:
        rewards = dataset["rewards"].flatten()
        ratios = np.exp(np.clip(eval_log_probs - behavior_log_probs, -max_ratio, max_ratio))
        unique_eps = np.unique(episode_ids)
        ep_values = []
        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            r_ep = rewards[mask]
            ratio_ep = ratios[mask]
            T = len(r_ep)
            # cumulative product of ratios
            cum_ratios = np.cumprod(ratio_ep)
            # discount factors
            gammas = discount ** np.arange(T)
            ep_value = np.sum(gammas * r_ep * cum_ratios)
            ep_values.append(ep_value)
        return float(np.mean(ep_values))


class WeightedImportanceSampling:
    """Self-normalized (weighted) importance sampling estimator (WIS)."""

    def estimate(
        self,
        dataset: dict,
        behavior_log_probs: np.ndarray,
        eval_log_probs: np.ndarray,
        episode_ids: np.ndarray,
        discount: float = 0.99,
        max_ratio: float = 20.0,
    ) -> float:
        rewards = dataset["rewards"].flatten()
        log_ratios = np.clip(eval_log_probs - behavior_log_probs, -max_ratio, max_ratio)
        ratios = np.exp(log_ratios)
        unique_eps = np.unique(episode_ids)
        ep_values = []
        ep_weights = []
        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            r_ep = rewards[mask]
            ratio_ep = ratios[mask]
            T = len(r_ep)
            cum_ratios = np.cumprod(ratio_ep)
            gammas = discount ** np.arange(T)
            ep_value = np.sum(gammas * r_ep * cum_ratios)
            ep_weight = cum_ratios[-1] if len(cum_ratios) > 0 else 1.0
            ep_values.append(ep_value)
            ep_weights.append(ep_weight)

        ep_values = np.array(ep_values)
        ep_weights = np.array(ep_weights)
        total_weight = ep_weights.sum()
        if total_weight < 1e-10:
            return float(np.mean(ep_values))
        return float((ep_values * ep_weights).sum() / total_weight)


class DoublyRobust:
    """Doubly robust OPE estimator."""

    def estimate(
        self,
        dataset: dict,
        behavior_log_probs: np.ndarray,
        eval_log_probs: np.ndarray,
        q_function,
        v_function,
        episode_ids: np.ndarray,
        discount: float = 0.99,
        max_ratio: float = 20.0,
    ) -> float:
        # Simplified DR: average of direct method (FQE) and WIS
        wis = WeightedImportanceSampling()
        wis_est = wis.estimate(
            dataset, behavior_log_probs, eval_log_probs, episode_ids, discount, max_ratio
        )
        # direct method estimate is the FQE value
        return float((q_function + wis_est) / 2)


def bootstrap_confidence_intervals(data: list, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Bootstrap CI for the mean.

    Returns (lower, upper) confidence interval.
    """
    if len(data) == 0:
        return (0.0, 0.0)
    data_arr = np.array(data)
    estimates = [
        np.mean(np.random.choice(data_arr, size=len(data_arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lower = float(np.percentile(estimates, alpha * 100))
    upper = float(np.percentile(estimates, (1 - alpha) * 100))
    return (lower, upper)


class OPEEvaluator:
    """Orchestrates offline policy evaluation using multiple methods."""

    def __init__(self, dataset: dict, behavior_policy=None, device: str = "cpu"):
        self.dataset = dataset
        self.behavior_policy = behavior_policy
        self.device = device

    def evaluate_policy(
        self,
        eval_policy,
        policy_name: str = "eval",
        methods: Optional[List[str]] = None,
        discount: float = 0.99,
    ) -> OPEResult:
        if methods is None:
            methods = ["fqe", "wis", "dr"]

        obs = self.dataset["observations"]
        actions = self.dataset["actions"]
        rewards = self.dataset["rewards"].flatten()
        episode_ids = self.dataset.get("episode_ids", np.zeros(len(rewards), dtype=np.int64))

        results = {}

        # FQE
        if "fqe" in methods:
            from offline_rl.evaluation.fqe import FQE, FQEConfig

            fqe_config = FQEConfig(n_iterations=20, batch_size=min(256, len(obs)))
            fqe = FQE(obs.shape[1], actions.shape[1], eval_policy, fqe_config, self.device)
            fqe.fit(self.dataset, n_iterations=20)
            results["fqe_estimate"] = fqe.estimate_return(obs[: min(1000, len(obs))])
        else:
            results["fqe_estimate"] = None

        # WIS and DR
        if "wis" in methods or "dr" in methods:
            eval_log_probs = self._compute_log_probs(eval_policy, obs, actions)
            behavior_log_probs = (
                self._compute_log_probs(self.behavior_policy, obs, actions)
                if self.behavior_policy is not None
                else np.zeros(len(obs))
            )

            wis = WeightedImportanceSampling()
            results["wis_estimate"] = wis.estimate(
                self.dataset, behavior_log_probs, eval_log_probs, episode_ids, discount
            )

            if "dr" in methods and results.get("fqe_estimate") is not None:
                dr = DoublyRobust()
                results["dr_estimate"] = dr.estimate(
                    self.dataset,
                    behavior_log_probs,
                    eval_log_probs,
                    q_function=results["fqe_estimate"],
                    v_function=results["wis_estimate"],
                    episode_ids=episode_ids,
                    discount=discount,
                )
            else:
                results["dr_estimate"] = None
        else:
            results["wis_estimate"] = None
            results["dr_estimate"] = None

        # Bootstrap CI on episode returns
        ep_returns = []
        unique_eps = np.unique(episode_ids)[:100]
        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            ep_returns.append(float(rewards[mask].sum()))

        ci = bootstrap_confidence_intervals(ep_returns) if ep_returns else (0.0, 0.0)

        return OPEResult(
            policy_name=policy_name,
            bootstrap_ci_95=ci,
            unsafe_action_rate=0.0,
            ood_action_fraction=0.0,
            **results,
        )

    def _compute_log_probs(self, policy, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute surrogate log probs assuming Gaussian noise around policy actions."""
        if policy is None:
            return np.zeros(len(obs))

        batch_size = 512
        log_probs = []
        for i in range(0, len(obs), batch_size):
            batch_obs = obs[i : i + batch_size]
            if hasattr(policy, "select_action"):
                pred_actions = np.array([policy.select_action(o) for o in batch_obs])
            elif callable(policy):
                pred_actions = np.array([policy(o) for o in batch_obs])
            else:
                pred_actions = np.zeros((len(batch_obs), actions.shape[1]))

            diff = actions[i : i + batch_size] - pred_actions
            act_dim = actions.shape[1]
            # Gaussian log prob with std=0.3
            lp = -0.5 * (diff**2 / 0.09).sum(axis=1) - act_dim * np.log(0.3 * np.sqrt(2 * np.pi))
            log_probs.append(lp)
        return np.concatenate(log_probs)


class ModelBasedEstimator:
    """Learn transition model, estimate return via short rollouts."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        from offline_rl.models.mlp import MLP

        self.device = device
        self.obs_dim = obs_dim
        self.transition_model = MLP(obs_dim + act_dim, obs_dim, hidden_dims).to(device)
        self.reward_model = MLP(obs_dim + act_dim, 1, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.transition_model.parameters()) + list(self.reward_model.parameters()),
            lr=1e-3,
        )
        self._fitted = False

    def fit_transition_model(
        self, dataset: dict, n_epochs: int = 10, batch_size: int = 256
    ) -> list:
        """Fit dynamics and reward models on dataset."""
        import torch.nn.functional as F

        obs = torch.FloatTensor(dataset["observations"]).to(self.device)
        actions = torch.FloatTensor(dataset["actions"]).to(self.device)
        next_obs = torch.FloatTensor(dataset["next_observations"]).to(self.device)
        rewards = torch.FloatTensor(dataset["rewards"]).to(self.device).unsqueeze(-1)

        sa = torch.cat([obs, actions], dim=-1)
        N = len(obs)
        losses = []

        for epoch in range(n_epochs):
            idx = torch.randperm(N)[:batch_size]
            pred_next = self.transition_model(sa[idx])
            pred_r = self.reward_model(sa[idx])
            loss = F.mse_loss(pred_next, next_obs[idx]) + F.mse_loss(pred_r, rewards[idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        self._fitted = True
        return losses

    def estimate(
        self,
        policy,
        n_rollouts: int = 100,
        horizon: int = 20,
        discount: float = 0.99,
    ) -> float:
        """Estimate policy return via model-based rollouts."""
        if not self._fitted:
            raise RuntimeError("Call fit_transition_model first")

        returns = []
        for _ in range(n_rollouts):
            obs = torch.FloatTensor(np.random.randn(self.obs_dim).astype(np.float32)).to(
                self.device
            )
            total_r = 0.0
            gamma = 1.0
            for t in range(horizon):
                obs_np = obs.cpu().numpy()
                if hasattr(policy, "select_action"):
                    action = policy.select_action(obs_np)
                else:
                    action = policy(obs_np)
                action_t = torch.FloatTensor(action).to(self.device)
                sa = torch.cat([obs, action_t])
                with torch.no_grad():
                    next_obs = self.transition_model(sa.unsqueeze(0)).squeeze(0)
                    reward = self.reward_model(sa.unsqueeze(0)).item()
                total_r += gamma * reward
                gamma *= discount
                obs = next_obs
            returns.append(total_r)
        return float(np.mean(returns))
