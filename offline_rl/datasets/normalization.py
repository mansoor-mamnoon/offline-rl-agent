"""Dataset normalization utilities."""

import json
import numpy as np
from pathlib import Path


class RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""

    def __init__(self, shape: tuple):
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape):
            x = x[np.newaxis]
        batch_size = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

        new_count = self.count + batch_size
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_size / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_size
        new_var = (m_a + m_b + delta**2 * self.count * batch_size / new_count) / new_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


def normalize_dataset(dataset: dict) -> tuple:
    """Normalize observations and rewards in a dataset.

    Returns (normalized_dataset, stats) where stats has:
    obs_mean, obs_std, reward_mean, reward_std
    """
    obs = dataset["observations"].astype(np.float64)
    rewards = dataset["rewards"].astype(np.float64)

    obs_mean = obs.mean(axis=0)
    obs_std = obs.std(axis=0) + 1e-8
    reward_mean = float(rewards.mean())
    reward_std = float(rewards.std() + 1e-8)

    stats = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }

    norm_dataset = dict(dataset)
    norm_dataset["observations"] = ((obs - obs_mean) / obs_std).astype(np.float32)
    norm_dataset["next_observations"] = (
        (dataset["next_observations"].astype(np.float64) - obs_mean) / obs_std
    ).astype(np.float32)
    norm_dataset["rewards"] = (
        (rewards - reward_mean) / reward_std
    ).astype(np.float32)

    return norm_dataset, stats


def denormalize_rewards(rewards: np.ndarray, stats: dict) -> np.ndarray:
    return (rewards * stats["reward_std"] + stats["reward_mean"]).astype(np.float32)


def save_normalization_stats(stats: dict, path: str):
    """Save normalization stats to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = float(v)
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_normalization_stats(path: str) -> dict:
    """Load normalization stats from a JSON file."""
    with open(path, "r") as f:
        raw = json.load(f)
    stats = {}
    for k, v in raw.items():
        if isinstance(v, list):
            stats[k] = np.array(v, dtype=np.float64)
        else:
            stats[k] = float(v)
    return stats
