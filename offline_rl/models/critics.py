"""Critic network implementations for offline RL."""

import torch
import torch.nn as nn
from typing import List
from offline_rl.models.mlp import MLP


class QNetwork(nn.Module):
    """Q-network: concatenates (obs, action) -> scalar Q value."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class DiscreteQNetwork(nn.Module):
    """Q-network for discrete actions: obs -> Q values for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = MLP(obs_dim, n_actions, hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DoubleQNetwork(nn.Module):
    """Twin Q-networks to reduce overestimation bias."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Returns (q1, q2)."""
        return self.q1(obs, action), self.q2(obs, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    """Value network: obs -> V(s)."""

    def __init__(self, obs_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = MLP(obs_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
