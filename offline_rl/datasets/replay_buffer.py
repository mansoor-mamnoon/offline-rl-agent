"""Replay buffer for offline RL dataset storage and sampling."""

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity replay buffer with optional prioritized sampling."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: str = "cpu",
        prioritized: bool = False,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.prioritized = prioritized

        self._observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)

        self._ptr = 0
        self._size = 0

        if prioritized:
            self._priorities = np.ones(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add a single transition."""
        idx = self._ptr % self.capacity
        self._observations[idx] = obs
        self._actions[idx] = np.asarray(action).flatten()[: self.act_dim]
        self._rewards[idx] = reward
        self._next_observations[idx] = next_obs
        self._dones[idx] = done

        if self.prioritized:
            max_prio = self._priorities[: self._size].max() if self._size > 0 else 1.0
            self._priorities[idx] = max_prio

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """Sample a batch. Returns dict of torch tensors on self.device."""
        if self._size < batch_size:
            batch_size = self._size

        if self.prioritized:
            probs = self._priorities[: self._size]
            probs = probs / probs.sum()
            indices = np.random.choice(self._size, size=batch_size, replace=False, p=probs)
        else:
            indices = np.random.randint(0, self._size, size=batch_size)

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        return {
            "observations": to_tensor(self._observations[indices]),
            "actions": to_tensor(self._actions[indices]),
            "rewards": to_tensor(self._rewards[indices]),
            "next_observations": to_tensor(self._next_observations[indices]),
            "dones": torch.tensor(self._dones[indices], dtype=torch.float32, device=self.device),
        }

    def load_from_dataset(self, dataset: dict):
        """Load all transitions from a dataset dict."""
        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        next_observations = dataset["next_observations"]
        dones = dataset["dones"]

        n = len(observations)

        # Handle discrete actions (1D integer array) -> 2D float
        if actions.ndim == 1:
            actions_2d = actions.reshape(-1, 1).astype(np.float32)
        else:
            actions_2d = actions.astype(np.float32)

        # Fill buffer (truncate to capacity if needed)
        n_to_load = min(n, self.capacity)
        self._observations[:n_to_load] = observations[:n_to_load].astype(np.float32)
        self._actions[:n_to_load] = actions_2d[:n_to_load, : self.act_dim]
        self._rewards[:n_to_load] = rewards[:n_to_load].astype(np.float32)
        self._next_observations[:n_to_load] = next_observations[:n_to_load].astype(np.float32)
        self._dones[:n_to_load] = dones[:n_to_load]

        if self.prioritized:
            self._priorities[:n_to_load] = 1.0

        self._ptr = n_to_load % self.capacity
        self._size = n_to_load

    def __len__(self) -> int:
        return self._size
