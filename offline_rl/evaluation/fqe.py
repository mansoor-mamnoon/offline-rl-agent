"""Fitted Q-Evaluation (FQE) for offline policy evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from offline_rl.models.critics import QNetwork
from offline_rl.models.mlp import polyak_update


@dataclass
class FQEConfig:
    n_iterations: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    discount: float = 0.99
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    tau: float = 0.005


class FQE:
    """Fitted Q-Evaluation.

    Iteratively fits a Q-function to estimate the value of a target policy
    using offline data.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy,
        config: Optional[FQEConfig] = None,
        device: str = "cpu",
    ):
        if config is None:
            config = FQEConfig()
        self.config = config
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.policy = policy

        self.q_network = QNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.q_target = QNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.lr)

    def _get_policy_action(self, obs_np: np.ndarray) -> np.ndarray:
        """Get action from policy given numpy observation batch."""
        if hasattr(self.policy, "select_action"):
            # Batch process
            actions = np.array([self.policy.select_action(o) for o in obs_np])
        elif callable(self.policy):
            actions = np.array([self.policy(o) for o in obs_np])
        else:
            raise ValueError("policy must be callable or have select_action method")
        return actions

    def fit(self, dataset: dict, n_iterations: Optional[int] = None) -> list:
        """Fit Q-function to the dataset.

        Returns list of loss values per iteration.
        """
        if n_iterations is None:
            n_iterations = self.config.n_iterations

        obs = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"].flatten()
        next_obs = dataset["next_observations"]
        dones = dataset["dones"].flatten().astype(np.float32)

        n = len(obs)
        losses = []

        for _ in range(n_iterations):
            # Sample mini-batch
            idx = np.random.randint(0, n, size=self.config.batch_size)
            obs_b = torch.FloatTensor(obs[idx]).to(self.device)
            actions_b = torch.FloatTensor(
                actions[idx] if actions.ndim > 1 else actions[idx].reshape(-1, 1)
            ).to(self.device)
            rewards_b = torch.FloatTensor(rewards[idx]).to(self.device)
            next_obs_b = torch.FloatTensor(next_obs[idx]).to(self.device)
            dones_b = torch.FloatTensor(dones[idx]).to(self.device)

            with torch.no_grad():
                next_actions_np = self._get_policy_action(next_obs[idx])
                next_actions_b = torch.FloatTensor(next_actions_np).to(self.device)
                if next_actions_b.ndim == 1:
                    next_actions_b = next_actions_b.unsqueeze(1)
                q_next = self.q_target(next_obs_b, next_actions_b)
                target = rewards_b + self.config.discount * (1 - dones_b) * q_next

            q_pred = self.q_network(obs_b, actions_b)
            loss = F.mse_loss(q_pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            polyak_update(self.q_network, self.q_target, self.config.tau)
            losses.append(loss.item())

        return losses

    def estimate_return(self, states: Optional[np.ndarray] = None) -> float:
        """Estimate expected return via Q(s, pi(s)).mean()."""
        if states is None:
            raise ValueError("states must be provided")
        actions_np = self._get_policy_action(states)
        obs_t = torch.FloatTensor(states).to(self.device)
        act_t = torch.FloatTensor(actions_np).to(self.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(1)
        with torch.no_grad():
            q_vals = self.q_network(obs_t, act_t)
        return float(q_vals.mean().item())

    def get_q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return Q-values for given state-action pairs."""
        obs_t = torch.FloatTensor(states).to(self.device)
        act_t = torch.FloatTensor(actions).to(self.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(1)
        with torch.no_grad():
            q_vals = self.q_network(obs_t, act_t)
        return q_vals.cpu().numpy()
