"""Dataset loading, validation, and splitting utilities."""

import numpy as np
import h5py

REQUIRED_KEYS = {"observations", "actions", "rewards", "next_observations", "dones"}


def load_dataset(path: str) -> dict:
    """Load a dataset from HDF5 (.h5/.hdf5) or NumPy (.npz) file.

    Returns dict with keys: observations, actions, rewards, next_observations,
    dones, episode_ids.
    """
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=False)
        dataset = {k: data[k] for k in data.files}
    elif path.endswith(".h5") or path.endswith(".hdf5"):
        dataset = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                dataset[key] = f[key][:]
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .h5, .hdf5, or .npz")

    # Ensure episode_ids exists
    if "episode_ids" not in dataset:
        n = len(dataset["observations"])
        dataset["episode_ids"] = np.zeros(n, dtype=np.int64)

    # Ensure consistent dtypes
    dataset["observations"] = dataset["observations"].astype(np.float32)
    dataset["rewards"] = dataset["rewards"].astype(np.float32)
    dataset["next_observations"] = dataset["next_observations"].astype(np.float32)
    dataset["dones"] = dataset["dones"].astype(bool)
    dataset["episode_ids"] = dataset["episode_ids"].astype(np.int64)

    return dataset


def validate_dataset(dataset: dict) -> bool:
    """Validate dataset shapes and values. Returns True if valid."""
    # Check required keys
    missing = REQUIRED_KEYS - set(dataset.keys())
    if missing:
        raise ValueError(f"Dataset missing required keys: {missing}")

    n = len(dataset["observations"])

    # Check all arrays have same length
    for key in REQUIRED_KEYS:
        if len(dataset[key]) != n:
            raise ValueError(f"Key '{key}' has length {len(dataset[key])}, expected {n}")

    # Check for NaN/inf
    for key in ["observations", "rewards", "next_observations"]:
        arr = dataset[key]
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN values found in '{key}'")
        if np.any(np.isinf(arr)):
            raise ValueError(f"Inf values found in '{key}'")

    return True


def split_dataset(dataset: dict, train_frac: float = 0.9) -> tuple:
    """Split dataset into train/val by episode boundaries.

    Returns (train_dataset, val_dataset).
    """
    episode_ids = dataset["episode_ids"]
    unique_eps = np.unique(episode_ids)
    n_train_eps = max(1, int(len(unique_eps) * train_frac))

    train_eps = set(unique_eps[:n_train_eps])
    val_eps = set(unique_eps[n_train_eps:])

    train_mask = np.array([eid in train_eps for eid in episode_ids])
    val_mask = np.array([eid in val_eps for eid in episode_ids])

    def filter_dataset(mask: np.ndarray) -> dict:
        return {k: v[mask] for k, v in dataset.items()}

    return filter_dataset(train_mask), filter_dataset(val_mask)


def dataset_info(dataset: dict) -> dict:
    """Return summary statistics about the dataset."""
    n_transitions = len(dataset["observations"])
    episode_ids = dataset.get("episode_ids", np.zeros(n_transitions))
    n_episodes = len(np.unique(episode_ids))
    obs = dataset["observations"]
    act = dataset["actions"]
    rewards = dataset["rewards"]

    return {
        "n_transitions": n_transitions,
        "n_episodes": n_episodes,
        "obs_dim": obs.shape[1] if obs.ndim > 1 else 1,
        "act_dim": act.shape[1] if act.ndim > 1 else 1,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
    }
