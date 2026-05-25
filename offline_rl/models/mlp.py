"""MLP and ensemble neural network implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _build_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations.keys())}")
    return activations[name]


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer norm and dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(_build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation is not None:
            layers.append(_build_activation(output_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleMLP(nn.Module):
    """Ensemble of MLPs for uncertainty estimation."""

    def __init__(
        self,
        n_ensemble: int,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.n_ensemble = n_ensemble
        self.models = nn.ModuleList(
            [MLP(input_dim, output_dim, hidden_dims) for _ in range(n_ensemble)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (n_ensemble, batch, output_dim)."""
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs, dim=0)


class GaussianMLP(nn.Module):
    """MLP that outputs (mean, log_std) for a Gaussian distribution."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Shared trunk
        trunk_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, h))
            trunk_layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.mean_head = nn.Linear(in_dim, output_dim)
        self.log_std_head = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """Returns (mean, log_std). log_std is clamped to [-20, 2]."""
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def log_prob(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under Gaussian."""
        mean, log_std = self(x)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1)

    def sample(self, x: torch.Tensor):
        """Sample action and return (action, log_prob)."""
        mean, log_std = self(x)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


def polyak_update(source: nn.Module, target: nn.Module, tau: float):
    """Soft update: target = tau * source + (1 - tau) * target."""
    with torch.no_grad():
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.mul_(1.0 - tau)
            t_param.data.add_(tau * s_param.data)


def get_parameter_count(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, path: str):
    """Save model state dict."""
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str) -> nn.Module:
    """Load model state dict in-place and return model."""
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model
