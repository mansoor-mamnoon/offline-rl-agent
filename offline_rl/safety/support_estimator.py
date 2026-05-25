"""k-NN based support estimator for detecting OOD state-action pairs."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SupportEstimator:
    """Estimates whether a (state, action) pair is in the dataset support.

    Uses k-nearest neighbor distances in the normalized state-action space.
    """

    def __init__(self, method: str = "knn", k: int = 10):
        self.k = k
        self.method = method
        self.knn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
        self._fitted = False
        self._scale = None
        self._mean = None
        self._ref_dist = None

    def fit(self, dataset_obs: np.ndarray, dataset_actions: np.ndarray):
        """Fit the support estimator on dataset observations and actions."""
        if dataset_obs.ndim == 1:
            dataset_obs = dataset_obs.reshape(-1, 1)
        if dataset_actions.ndim == 1:
            dataset_actions = dataset_actions.reshape(-1, 1)

        sa = np.concatenate([dataset_obs, dataset_actions], axis=1).astype(np.float32)
        self._mean = sa.mean(axis=0)
        self._scale = sa.std(axis=0) + 1e-8
        sa_norm = (sa - self._mean) / self._scale

        self.knn.fit(sa_norm)

        # Compute reference distance from in-distribution data
        n_ref = min(1000, len(sa_norm))
        dists, _ = self.knn.kneighbors(sa_norm[:n_ref])
        self._ref_dist = float(dists[:, 1:].mean())  # skip self (index 0)
        self._fitted = True

    def support_score(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute support score for a single (obs, action) pair.

        Returns score in [0, 1], higher = more in-support.
        """
        assert self._fitted, "Call fit() before support_score()"
        sa = np.concatenate([obs.flatten(), action.flatten()]).astype(np.float32)
        sa_norm = (sa - self._mean) / self._scale
        dists, _ = self.knn.kneighbors(sa_norm.reshape(1, -1))
        # Use neighbors 1: (not self) if shape allows, else use 0:
        if dists.shape[1] > 1:
            mean_dist = float(dists[0, 1:].mean())
        else:
            mean_dist = float(dists[0, 0])
        return float(1.0 / (1.0 + mean_dist / (self._ref_dist + 1e-8)))

    def batch_support_scores(self, obs_batch: np.ndarray, action_batch: np.ndarray) -> np.ndarray:
        """Compute support scores for a batch of (obs, action) pairs."""
        assert self._fitted, "Call fit() before batch_support_scores()"
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(-1, 1)
        if action_batch.ndim == 1:
            action_batch = action_batch.reshape(-1, 1)
        sa = np.concatenate([obs_batch, action_batch], axis=1).astype(np.float32)
        sa_norm = (sa - self._mean) / self._scale
        dists, _ = self.knn.kneighbors(sa_norm)
        if dists.shape[1] > 1:
            mean_dists = dists[:, 1:].mean(axis=1)
        else:
            mean_dists = dists[:, 0]
        return 1.0 / (1.0 + mean_dists / (self._ref_dist + 1e-8))

    def is_in_support(self, obs: np.ndarray, action: np.ndarray, threshold: float = 0.3) -> bool:
        """Return True if (obs, action) is in support above threshold."""
        return self.support_score(obs, action) >= threshold
