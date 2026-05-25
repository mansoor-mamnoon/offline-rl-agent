"""Advantage Weighted Actor Critic (AWAC) for offline RL.

AWAC bridges imitation learning and value-based RL by weighting actor
updates by the exponential advantage, similar to IQL's policy extraction
but with a separate Q-function training step.

Reference: Nair et al., "Accelerating Online Reinforcement Learning with
Offline Datasets" (2021).
"""

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from offline_rl.models.mlp import GaussianMLP, polyak_update
from offline_rl.models.critics import DoubleQNetwork, ValueNetwork


@dataclass
class AWACConfig:
    hidden_dims: list = field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    beta: float = 1.0  # advantage temperature (lower = more conservative)
    discount: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    n_action_samples: int = 16  # actions sampled for advantage estimation
    n_epochs: int = 100


class AWAC:
    """Advantage Weighted Actor Critic."""

    def __init__(self, obs_dim: int, act_dim: int, config: AWACConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.qf = DoubleQNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.qf_target = DoubleQNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.qf_target.load_state_dict(self.qf.state_dict())

        self.policy = GaussianMLP(obs_dim, act_dim, config.hidden_dims).to(device)
        self.vf = ValueNetwork(obs_dim, config.hidden_dims).to(device)

        self.qf_optimizer = torch.optim.Adam(self.qf.parameters(), lr=config.lr)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=config.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        self._n_steps = 0

    def _sample_policy(self, obs):
        mean, log_std = self.policy(obs)
        std = log_std.exp().clamp(0.01, 2.0)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob, mean

    def train_step(self, batch: dict) -> dict:
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Value function: V(s) approx E_a~pi[Q(s,a)]
        with torch.no_grad():
            pi_actions, _, _ = self._sample_policy(obs)
            q1_pi, q2_pi = self.qf_target(obs, pi_actions)
            v_target = torch.min(q1_pi, q2_pi)

        vf_loss = F.mse_loss(self.vf(obs), v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        # Q function: standard Bellman backup using V(s')
        with torch.no_grad():
            next_v = self.vf(next_obs)
            q_target = rewards + self.config.discount * (1 - dones) * next_v

        q1, q2 = self.qf(obs, actions)
        qf_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # Policy: advantage-weighted regression
        with torch.no_grad():
            q1_data, q2_data = self.qf(obs, actions)
            q_data = torch.min(q1_data, q2_data)
            v_data = self.vf(obs)
            advantage = q_data - v_data
            # Exponential advantage weighting (clipped for stability)
            weights = torch.exp(advantage / self.config.beta).clamp(0, 100.0)
            weights = weights / (weights.mean() + 1e-8)

        # Log-prob of dataset actions under current policy
        mean, log_std = self.policy(obs)
        std = log_std.exp().clamp(0.01, 2.0)
        dist = torch.distributions.Normal(mean, std)
        # Atanh to get pre-tanh action
        actions_clamped = actions.clamp(-0.999, 0.999)
        raw_actions = torch.atanh(actions_clamped)
        log_prob = dist.log_prob(raw_actions).sum(-1, keepdim=True)

        policy_loss = -(weights * log_prob).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        polyak_update(self.qf, self.qf_target, self.config.tau)
        self._n_steps += 1

        return {
            "qf_loss": qf_loss.item(),
            "vf_loss": vf_loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_advantage": advantage.mean().item(),
            "mean_weight": weights.mean().item(),
        }

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mean, log_std = self.policy(obs_t)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                action = torch.tanh(mean + std * torch.randn_like(std))
        return action.cpu().numpy().squeeze()

    def save(self, path: str):
        torch.save(
            {
                "qf": self.qf.state_dict(),
                "qf_target": self.qf_target.state_dict(),
                "policy": self.policy.state_dict(),
                "vf": self.vf.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AWAC":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["config"], device)
        agent.qf.load_state_dict(ckpt["qf"])
        agent.qf_target.load_state_dict(ckpt["qf_target"])
        agent.policy.load_state_dict(ckpt["policy"])
        agent.vf.load_state_dict(ckpt["vf"])
        return agent
