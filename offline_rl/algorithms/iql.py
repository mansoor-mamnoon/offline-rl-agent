"""Implicit Q-Learning (IQL) for offline RL."""

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List

from offline_rl.models.mlp import GaussianMLP, polyak_update
from offline_rl.models.critics import DoubleQNetwork, ValueNetwork


@dataclass
class IQLConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    tau: float = 0.7  # expectile for value loss
    beta: float = 3.0  # advantage temperature for policy loss
    discount: float = 0.99
    lr: float = 3e-4
    tau_soft: float = 0.005
    batch_size: int = 256
    n_epochs: int = 100


class IQL:
    """Implicit Q-Learning agent."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: IQLConfig = None,
        device: str = "cpu",
    ):
        if config is None:
            config = IQLConfig()
        self.config = config
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Networks
        self.qf = DoubleQNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.qf_target = DoubleQNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.qf_target.load_state_dict(self.qf.state_dict())

        self.vf = ValueNetwork(obs_dim, config.hidden_dims).to(device)
        self.policy = GaussianMLP(obs_dim, act_dim, config.hidden_dims).to(device)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=config.lr)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=config.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        self._n_train_steps = 0

    def _expectile_loss(self, diff: torch.Tensor, tau: float) -> torch.Tensor:
        """Asymmetric L2 loss: L_tau(diff) = |tau - 1(diff<0)| * diff^2."""
        weight = torch.where(
            diff > 0,
            tau * torch.ones_like(diff),
            (1 - tau) * torch.ones_like(diff),
        )
        return (weight * diff.pow(2)).mean()

    def train_step(self, batch: dict) -> dict:
        obs = batch["observations"].to(self.device).float()
        actions = batch["actions"].to(self.device).float()
        rewards = batch["rewards"].to(self.device).float()
        next_obs = batch["next_observations"].to(self.device).float()
        dones = batch["dones"].to(self.device).float()

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)
        if dones.ndim == 1:
            dones = dones.unsqueeze(1)

        # ── Value loss (expectile regression) ─────────────────────────────────
        with torch.no_grad():
            q1_target, q2_target = self.qf_target(obs, actions)
            q_target = torch.min(q1_target, q2_target).unsqueeze(1)

        v = self.vf(obs).unsqueeze(1)
        value_loss = self._expectile_loss(q_target - v, self.config.tau)

        self.v_optimizer.zero_grad()
        value_loss.backward()
        self.v_optimizer.step()

        # ── Q loss ────────────────────────────────────────────────────────────
        with torch.no_grad():
            v_next = self.vf(next_obs).unsqueeze(1)
            target_q = rewards + self.config.discount * (1 - dones) * v_next

        q1, q2 = self.qf(obs, actions)
        q1 = q1.unsqueeze(1)
        q2 = q2.unsqueeze(1)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ── Policy loss (advantage-weighted regression) ────────────────────────
        with torch.no_grad():
            q1_d, q2_d = self.qf(obs, actions)
            adv = torch.min(q1_d, q2_d) - self.vf(obs)
            weight = torch.exp(self.config.beta * adv).clamp(max=100.0)

        mean, log_std = self.policy(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_pi = dist.log_prob(actions).sum(-1)
        policy_loss = -(weight * log_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ── Target network update ─────────────────────────────────────────────
        polyak_update(self.qf, self.qf_target, self.config.tau_soft)

        self._n_train_steps += 1

        return {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_v": v.mean().item(),
            "mean_q": ((q1 + q2) / 2).mean().item(),
        }

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mean, log_std = self.policy(obs_t)
            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                action = mean + std * torch.randn_like(std)
        return action.cpu().numpy().squeeze()

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "qf": self.qf.state_dict(),
                "qf_target": self.qf_target.state_dict(),
                "vf": self.vf.state_dict(),
                "policy": self.policy.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "IQL":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["config"], device)
        agent.qf.load_state_dict(ckpt["qf"])
        agent.qf_target.load_state_dict(ckpt["qf_target"])
        agent.vf.load_state_dict(ckpt["vf"])
        agent.policy.load_state_dict(ckpt["policy"])
        return agent
