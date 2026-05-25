import numpy as np
import torch


class TrajectoryReplayBuffer:
    """Stores full trajectories and samples fixed-length subsequences."""

    def __init__(self, context_len: int = 20, device: str = "cpu"):
        self.context_len = context_len
        self.device = device
        self.trajectories = []  # list of episode dicts

    def load_from_dataset(self, dataset: dict):
        """Split dataset into per-episode trajectories."""
        episode_ids = dataset.get("episode_ids", np.zeros(len(dataset["rewards"])))
        unique_eps = np.unique(episode_ids)

        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            traj = {
                "observations": dataset["observations"][mask],
                "actions": dataset["actions"][mask],
                "rewards": dataset["rewards"][mask],
                "dones": dataset["dones"][mask],
            }
            # Compute returns-to-go
            rewards = traj["rewards"].flatten()
            rtgs = np.zeros_like(rewards)
            running = 0
            for i in reversed(range(len(rewards))):
                running = rewards[i] + 0.99 * running
                rtgs[i] = running
            traj["returns_to_go"] = rtgs
            traj["timesteps"] = np.arange(len(rewards))
            self.trajectories.append(traj)

    def sample(self, batch_size: int) -> dict:
        batch = {k: [] for k in ["observations", "actions", "returns_to_go", "timesteps"]}

        for _ in range(batch_size):
            traj = self.trajectories[np.random.randint(len(self.trajectories))]
            T = len(traj["rewards"])

            if T <= self.context_len:
                start = 0
                end = T
            else:
                start = np.random.randint(0, T - self.context_len)
                end = start + self.context_len

            obs = traj["observations"][start:end]
            acts = traj["actions"][start:end]
            rtgs = traj["returns_to_go"][start:end]
            ts = traj["timesteps"][start:end]

            # Pad to context_len if shorter
            pad_len = self.context_len - len(obs)
            if pad_len > 0:
                obs = np.concatenate([np.zeros((pad_len, obs.shape[1])), obs])
                acts = np.concatenate([np.zeros((pad_len, acts.shape[1])), acts])
                rtgs = np.concatenate([np.zeros(pad_len), rtgs])
                ts = np.concatenate([np.zeros(pad_len, dtype=int), ts])

            batch["observations"].append(obs)
            batch["actions"].append(acts)
            batch["returns_to_go"].append(rtgs)
            batch["timesteps"].append(ts)

        return {
            "observations": torch.FloatTensor(np.array(batch["observations"])).to(self.device),
            "actions": torch.FloatTensor(np.array(batch["actions"])).to(self.device),
            "returns_to_go": torch.FloatTensor(np.array(batch["returns_to_go"])).unsqueeze(-1).to(self.device),
            "timesteps": torch.LongTensor(np.array(batch["timesteps"])).to(self.device),
        }

    def __len__(self):
        return len(self.trajectories)
