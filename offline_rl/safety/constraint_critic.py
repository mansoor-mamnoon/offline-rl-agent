import torch
import torch.nn as nn
import numpy as np
from offline_rl.models.mlp import MLP


class ConstraintCritic:
    """Learns cost Q-functions for multiple constraints."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_constraints: int = 3,
        hidden_dims: list = None,
        lr: float = 1e-3,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.n_constraints = n_constraints
        self.device = device
        self.discount = discount

        # One Q network per constraint
        self.q_networks = nn.ModuleList(
            [MLP(obs_dim + act_dim, 1, hidden_dims) for _ in range(n_constraints)]
        ).to(device)

        self.q_targets = nn.ModuleList(
            [MLP(obs_dim + act_dim, 1, hidden_dims) for _ in range(n_constraints)]
        ).to(device)

        # Copy weights to targets
        for q, qt in zip(self.q_networks, self.q_targets):
            qt.load_state_dict(q.state_dict())

        self.optimizer = torch.optim.Adam(self.q_networks.parameters(), lr=lr)

        # Constraint names for logging
        self.constraint_names = [
            "slo_latency_violation",
            "overload_violation",
            "error_rate_violation",
        ][:n_constraints]

    def train_step(self, batch: dict) -> dict:
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        next_obs = batch["next_observations"].to(self.device)
        next_actions = batch.get("next_actions", torch.randn_like(actions)).to(self.device)
        dones = batch["dones"].to(self.device)

        # Costs: from batch or generate from obs/next_obs (simplified)
        if "costs" in batch:
            costs = batch["costs"].to(self.device)  # (B, n_constraints)
        else:
            # Synthetic costs based on obs features (for testing)
            costs = (torch.randn(obs.shape[0], self.n_constraints) > 1.5).float().to(self.device)

        sa = torch.cat([obs, actions], dim=-1)
        next_sa = torch.cat([next_obs, next_actions], dim=-1)

        total_loss = 0
        losses = {}

        for i, (q_net, q_tgt, name) in enumerate(
            zip(self.q_networks, self.q_targets, self.constraint_names)
        ):
            target = costs[:, i : i + 1] + self.discount * (1 - dones) * q_tgt(next_sa)
            q_pred = q_net(sa)
            loss = nn.functional.mse_loss(q_pred, target.detach())
            total_loss += loss
            losses[f"cost_loss_{name}"] = loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return losses

    def estimate_constraint_violation(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        act_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        sa = torch.cat([obs_t, act_t], dim=-1)

        with torch.no_grad():
            violations = np.array([q(sa).item() for q in self.q_networks])
        return np.clip(violations, 0, 1)

    def is_safe(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        thresholds: list = None,
    ) -> bool:
        if thresholds is None:
            thresholds = [0.1] * self.n_constraints
        violations = self.estimate_constraint_violation(obs, action)
        return all(v < t for v, t in zip(violations, thresholds))
