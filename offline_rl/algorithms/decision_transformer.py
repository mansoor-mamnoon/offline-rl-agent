from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from offline_rl.models.transformer import DecisionTransformerModel


@dataclass
class DTConfig:
    context_len: int = 20
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 1
    lr: float = 1e-4
    batch_size: int = 64
    target_return: float = 90.0
    dropout: float = 0.1
    max_ep_len: int = 1000
    n_epochs: int = 100


class DecisionTransformer:
    def __init__(self, obs_dim: int, act_dim: int, config: DTConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.model = DecisionTransformerModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            context_len=config.context_len,
            max_ep_len=config.max_ep_len,
            dropout=config.dropout,
        ).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-4)

    def train_step(self, batch: dict) -> dict:
        states = batch["observations"]  # (B, T, obs_dim)
        actions = batch["actions"]  # (B, T, act_dim)
        rtgs = batch["returns_to_go"]  # (B, T, 1)
        timesteps = batch["timesteps"]  # (B, T)

        pred_actions = self.model(states, actions, rtgs, timesteps)

        # MSE loss on all timesteps (teacher forcing)
        loss = nn.functional.mse_loss(pred_actions, actions)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        return {"action_loss": loss.item()}

    def select_action(
        self,
        obs_history: np.ndarray,  # (T, obs_dim)
        action_history: np.ndarray,  # (T, act_dim)
        rtg_history: np.ndarray,  # (T,)
        timestep: int,
        target_return: float = None,
    ) -> np.ndarray:
        if target_return is None:
            target_return = self.config.target_return

        T = min(len(obs_history), self.config.context_len)
        obs = obs_history[-T:]
        acts = action_history[-T:]
        rtgs = rtg_history[-T:].reshape(-1, 1)
        ts = np.arange(timestep - T + 1, timestep + 1).clip(0, self.config.max_ep_len - 1)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        acts_t = torch.FloatTensor(acts).unsqueeze(0).to(self.device)
        rtgs_t = torch.FloatTensor(rtgs).unsqueeze(0).to(self.device)
        ts_t = torch.LongTensor(ts).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(obs_t, acts_t, rtgs_t, ts_t)

        return pred[0, -1].cpu().numpy()

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DecisionTransformer":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["config"], device)
        agent.model.load_state_dict(ckpt["model"])
        return agent
