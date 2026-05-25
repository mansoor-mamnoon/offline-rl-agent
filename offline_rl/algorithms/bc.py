"""Behavior Cloning algorithm for offline RL."""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from offline_rl.models.mlp import MLP, GaussianMLP


@dataclass
class BCConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 100
    l2_reg: float = 1e-5
    action_space: str = "continuous"  # "continuous" or "discrete"


class BehaviorCloning:
    """Behavior Cloning (supervised imitation learning)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: Optional[BCConfig] = None,
        device: str = "cpu",
    ):
        if config is None:
            config = BCConfig()
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        if config.action_space == "continuous":
            self.policy = GaussianMLP(obs_dim, act_dim, config.hidden_dims).to(device)
        else:
            # Discrete: obs -> logits for each action
            self.policy = MLP(obs_dim, act_dim, config.hidden_dims).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr, weight_decay=config.l2_reg
        )

    def train_step(self, batch: dict) -> dict:
        """Single gradient step.

        Args:
            batch: dict with 'observations' and 'actions' tensors.

        Returns:
            dict with 'loss' and 'nll'.
        """
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)

        self.optimizer.zero_grad()

        if self.config.action_space == "continuous":
            mean, log_std = self.policy(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            nll = -dist.log_prob(actions).sum(dim=-1).mean()
            loss = nll
        else:
            logits = self.policy(obs)
            # actions may be (batch, 1) float or (batch,) int
            if actions.ndim == 2:
                actions_int = actions.squeeze(-1).long()
            else:
                actions_int = actions.long()
            nll = F.cross_entropy(logits, actions_int)
            loss = nll

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "nll": nll.item()}

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action given observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            if self.config.action_space == "continuous":
                mean, log_std = self.policy(obs_t)
                if deterministic:
                    action = mean
                else:
                    std = torch.exp(log_std)
                    action = torch.distributions.Normal(mean, std).sample()
                return action.squeeze(0).cpu().numpy()
            else:
                logits = self.policy(obs_t)
                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).squeeze(-1)
                return action.squeeze(0).cpu().numpy()

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BehaviorCloning":
        """Load a saved BehaviorCloning checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        obs_dim = checkpoint["obs_dim"]
        act_dim = checkpoint["act_dim"]
        agent = cls(obs_dim, act_dim, config, device)
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return agent
