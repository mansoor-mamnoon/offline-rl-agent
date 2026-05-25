"""TD3+BC: TD3 with Behavior Cloning regularization for offline RL."""

from dataclasses import dataclass, field
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List

from offline_rl.models.mlp import MLP, polyak_update
from offline_rl.models.critics import DoubleQNetwork


@dataclass
class TD3BCConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    alpha: float = 2.5       # BC regularization weight
    tau: float = 0.005
    discount: float = 0.99
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    batch_size: int = 256
    n_epochs: int = 100


class TD3BC:
    """TD3+BC: TD3 with Behavior Cloning loss for offline RL.

    Policy loss: -lambda * Q(s, pi(s)) + (pi(s) - a)^2
    where lambda = alpha / mean(|Q(s,a)|)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: TD3BCConfig = None,
        device: str = "cpu",
    ):
        if config is None:
            config = TD3BCConfig()
        self.config = config
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Deterministic policy
        self.actor = MLP(
            obs_dim, act_dim, config.hidden_dims, output_activation="tanh"
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        # Q networks
        self.qf = DoubleQNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.qf_target = DoubleQNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.qf_target.load_state_dict(self.qf.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=config.lr)

        self._n_train_steps = 0

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

        # ── Q loss ────────────────────────────────────────────────────────────
        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.config.policy_noise
            ).clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

            q1_next, q2_next = self.qf_target(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next).unsqueeze(1)
            target_q = rewards + self.config.discount * (1 - dones) * min_q_next

        q1, q2 = self.qf(obs, actions)
        q1 = q1.unsqueeze(1)
        q2 = q2.unsqueeze(1)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        policy_loss_val = 0.0
        bc_loss_val = 0.0
        lambda_val = 0.0

        # ── Policy loss (every policy_freq steps) ─────────────────────────────
        if self._n_train_steps % self.config.policy_freq == 0:
            pi = self.actor(obs)
            q1_pi, _ = self.qf(obs, pi)

            # Normalize: lambda = alpha / mean(|Q(s,a)|)
            lam = self.config.alpha / (q1_pi.abs().mean().detach() + 1e-8)
            lambda_val = lam.item()

            bc_loss = F.mse_loss(pi, actions)
            policy_loss = -lam * q1_pi.mean() + bc_loss

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            policy_loss_val = policy_loss.item()
            bc_loss_val = bc_loss.item()

            # Target network updates
            polyak_update(self.actor, self.actor_target, self.config.tau)
            polyak_update(self.qf, self.qf_target, self.config.tau)

        self._n_train_steps += 1

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss_val,
            "bc_loss": bc_loss_val,
            "lambda_val": lambda_val,
        }

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_t)
            if not deterministic:
                noise = torch.randn_like(action) * 0.1
                action = (action + noise).clamp(-1.0, 1.0)
        return action.cpu().numpy().squeeze()

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "qf": self.qf.state_dict(),
                "qf_target": self.qf_target.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TD3BC":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["config"], device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.actor_target.load_state_dict(ckpt["actor_target"])
        agent.qf.load_state_dict(ckpt["qf"])
        agent.qf_target.load_state_dict(ckpt["qf_target"])
        return agent
