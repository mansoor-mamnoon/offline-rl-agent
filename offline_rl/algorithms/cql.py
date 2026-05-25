"""Conservative Q-Learning (CQL) for offline RL."""

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List

from offline_rl.models.mlp import GaussianMLP, polyak_update
from offline_rl.models.critics import DoubleQNetwork


@dataclass
class CQLConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    alpha: float = 5.0
    alpha_auto: bool = True
    target_action_gap: float = -1.0
    tau: float = 0.005
    discount: float = 0.99
    n_random_actions: int = 10
    backup_entropy: bool = True
    batch_size: int = 256
    temp: float = 1.0
    n_epochs: int = 100


class CQL:
    """Conservative Q-Learning agent."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: CQLConfig = None,
        device: str = "cpu",
    ):
        if config is None:
            config = CQLConfig()
        self.config = config
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Q networks (double)
        self.qf = DoubleQNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.qf_target = DoubleQNetwork(obs_dim, act_dim, config.hidden_dims).to(device)
        self.qf_target.load_state_dict(self.qf.state_dict())

        # Policy (Gaussian)
        self.policy = GaussianMLP(obs_dim, act_dim, config.hidden_dims).to(device)

        # Optimizers
        self.qf_optimizer = torch.optim.Adam(self.qf.parameters(), lr=config.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        # Alpha (CQL regularization weight)
        self.log_alpha = torch.tensor(
            np.log(config.alpha),
            requires_grad=config.alpha_auto,
            device=device,
            dtype=torch.float32,
        )
        if config.alpha_auto:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr)

        # SAC temperature
        self.log_temp = torch.tensor(
            np.log(config.temp),
            requires_grad=True,
            device=device,
            dtype=torch.float32,
        )
        self.temp_optimizer = torch.optim.Adam([self.log_temp], lr=config.lr)

        self._n_train_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def temp(self):
        return self.log_temp.exp()

    def _sample_policy_actions(self, obs):
        mean, log_std = self.policy(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def train_step(self, batch: dict) -> dict:
        obs = batch["observations"].to(self.device).float()
        actions = batch["actions"].to(self.device).float()
        rewards = batch["rewards"].to(self.device).float()
        next_obs = batch["next_observations"].to(self.device).float()
        dones = batch["dones"].to(self.device).float()

        # Ensure rewards and dones are (N, 1)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)
        if dones.ndim == 1:
            dones = dones.unsqueeze(1)

        batch_size = obs.shape[0]

        with torch.no_grad():
            next_actions, next_log_pi = self._sample_policy_actions(next_obs)
            q1_next, q2_next = self.qf_target(next_obs, next_actions)
            q1_next = q1_next.unsqueeze(1)
            q2_next = q2_next.unsqueeze(1)
            min_q_next = torch.min(q1_next, q2_next)
            if self.config.backup_entropy:
                min_q_next = min_q_next - self.temp * next_log_pi
            target_q = rewards + self.config.discount * (1 - dones) * min_q_next

        # Q loss
        q1, q2 = self.qf(obs, actions)
        q1 = q1.unsqueeze(1)
        q2 = q2.unsqueeze(1)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # CQL conservative penalty
        # Sample random actions
        random_actions = (
            torch.FloatTensor(batch_size * self.config.n_random_actions, self.act_dim)
            .uniform_(-1, 1)
            .to(self.device)
        )
        obs_repeated = (
            obs.unsqueeze(1).repeat(1, self.config.n_random_actions, 1).view(-1, self.obs_dim)
        )
        next_obs_repeated = (
            next_obs.unsqueeze(1).repeat(1, self.config.n_random_actions, 1).view(-1, self.obs_dim)
        )

        with torch.no_grad():
            policy_actions, policy_log_pi = self._sample_policy_actions(obs_repeated)
            next_policy_actions, next_policy_log_pi = self._sample_policy_actions(next_obs_repeated)

        q1_rand, q2_rand = self.qf(obs_repeated, random_actions)
        q1_policy, q2_policy = self.qf(obs_repeated, policy_actions)
        q1_next_policy, q2_next_policy = self.qf(obs_repeated, next_policy_actions)

        # log pi for importance weighting
        random_log_pi = torch.log(
            torch.ones(batch_size * self.config.n_random_actions, 1, device=self.device)
            / (2**self.act_dim)
        )

        # reshape for cat
        def reshape_q(q):
            return q.view(batch_size, self.config.n_random_actions, 1)

        def reshape_lp(lp):
            return lp.view(batch_size, self.config.n_random_actions, 1)

        def reshape_rand_lp(lp):
            return lp.view(batch_size, self.config.n_random_actions, 1)

        cat_q1 = torch.cat(
            [
                reshape_q(q1_rand) - reshape_rand_lp(random_log_pi),
                reshape_q(q1_policy) - reshape_lp(policy_log_pi),
                reshape_q(q1_next_policy) - reshape_lp(next_policy_log_pi),
            ],
            dim=1,
        )
        cat_q2 = torch.cat(
            [
                reshape_q(q2_rand) - reshape_rand_lp(random_log_pi),
                reshape_q(q2_policy) - reshape_lp(policy_log_pi),
                reshape_q(q2_next_policy) - reshape_lp(next_policy_log_pi),
            ],
            dim=1,
        )

        cql_penalty = (torch.logsumexp(cat_q1, dim=1) - q1).mean() + (
            torch.logsumexp(cat_q2, dim=1) - q2
        ).mean()

        # Auto-tune alpha
        if self.config.alpha_auto:
            alpha_loss = -(self.log_alpha * (cql_penalty - self.config.target_action_gap).detach())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        total_q_loss = q_loss + self.alpha.detach() * cql_penalty

        self.qf_optimizer.zero_grad()
        total_q_loss.backward()
        self.qf_optimizer.step()

        # Policy loss (SAC-style)
        policy_actions_pi, log_pi = self._sample_policy_actions(obs)
        q1_pi, q2_pi = self.qf(obs, policy_actions_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.temp.detach() * log_pi.squeeze(1) - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Temperature loss
        temp_loss = -(self.log_temp * (log_pi.squeeze(1) + self.act_dim).detach()).mean()
        self.temp_optimizer.zero_grad()
        temp_loss.backward()
        self.temp_optimizer.step()

        # Polyak update
        polyak_update(self.qf, self.qf_target, self.config.tau)

        self._n_train_steps += 1

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "alpha": self.alpha.item(),
            "mean_q": ((q1 + q2) / 2).mean().item(),
            "ood_q": q1_rand.mean().item(),
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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "qf": self.qf.state_dict(),
                "qf_target": self.qf_target.state_dict(),
                "policy": self.policy.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CQL":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["config"], device)
        agent.qf.load_state_dict(ckpt["qf"])
        agent.qf_target.load_state_dict(ckpt["qf_target"])
        agent.policy.load_state_dict(ckpt["policy"])
        return agent
