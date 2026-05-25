"""Dataset diagnostics: coverage, entropy, OOD risk, outlier detection."""

from collections import Counter
from dataclasses import dataclass
from typing import Optional
import json

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class DiagnosticsReport:
    n_transitions: int
    n_episodes: int
    obs_dim: int
    act_dim: int
    coverage_score: float
    behavior_entropy: float
    reward_stats: dict
    episode_stats: dict
    ood_risk: dict
    action_coverage: dict
    outlier_indices: list
    state_pca_points: Optional[np.ndarray]  # (N, 2)
    pca_explained_variance: float
    support_mismatch_risk: float

    def to_dict(self) -> dict:
        d = {
            k: v
            for k, v in self.__dict__.items()
            if not isinstance(v, np.ndarray)
        }
        d["state_pca_points"] = (
            self.state_pca_points.tolist()
            if self.state_pca_points is not None
            else None
        )
        return d

    def summary_string(self) -> str:
        ood_level = self.ood_risk.get("risk_level", "unknown")
        ep_stats = self.episode_stats
        reward_skew = self.reward_stats.get("skewness", 0.0)

        warnings = []
        if ood_level == "high":
            frac = self.ood_risk.get("sparse_state_fraction", 0.0)
            warnings.append(f"{frac*100:.1f}% of random actions far from dataset support")
        if abs(reward_skew) > 2.0:
            warnings.append(f"Reward distribution heavily skewed (skewness={reward_skew:.2f})")
        if ep_stats.get("terminal_rate", 1.0) < 0.1:
            warnings.append("Very few episodes reach terminal state — check done flags")

        warn_str = "\n   · ".join(warnings) if warnings else "None"
        if warnings:
            warn_str = "· " + warn_str

        return (
            f"{'='*39}\n"
            f"Dataset Diagnostics\n"
            f"{'='*39}\n"
            f"Transitions:      {self.n_transitions:,}\n"
            f"Episodes:         {self.n_episodes:,}\n"
            f"Obs dim:          {self.obs_dim}\n"
            f"Act dim:          {self.act_dim}\n"
            f"\n"
            f"Coverage score:   {self.coverage_score:.3f}\n"
            f"Behavior entropy: {self.behavior_entropy:.3f}\n"
            f"OOD risk:         {ood_level}\n"
            f"Reward skew:      {reward_skew:.2f}\n"
            f"Terminal rate:    {ep_stats.get('terminal_rate', 0.0)*100:.1f}%\n"
            f"\n"
            f"Warnings:\n"
            f"   {warn_str}\n"
            f"{'='*39}"
        )


class DatasetDiagnostics:
    """Runs a battery of diagnostic checks on an offline RL dataset."""

    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.obs = dataset["observations"]
        self.actions = dataset["actions"]
        self.rewards = dataset["rewards"].flatten()
        self.dones = dataset["dones"].flatten()
        self.episode_ids = dataset.get(
            "episode_ids", np.zeros(len(self.rewards), dtype=np.int64)
        )

        # Ensure actions are 2D
        if self.actions.ndim == 1:
            self.actions = self.actions.reshape(-1, 1)

    def run_all(self) -> DiagnosticsReport:
        pca_result = self.compute_state_coverage_pca()
        return DiagnosticsReport(
            n_transitions=len(self.rewards),
            n_episodes=int(len(np.unique(self.episode_ids))),
            obs_dim=self.obs.shape[1] if self.obs.ndim > 1 else 1,
            act_dim=self.actions.shape[1],
            coverage_score=self.compute_coverage_score(),
            behavior_entropy=self.compute_behavior_entropy(),
            reward_stats=self.compute_reward_stats(),
            episode_stats=self.compute_episode_stats(),
            ood_risk=self.detect_ood_risk(),
            action_coverage=self.compute_action_coverage(),
            outlier_indices=self.detect_outlier_transitions(),
            state_pca_points=pca_result["state_pca_points"],
            pca_explained_variance=pca_result["pca_explained_variance"],
            support_mismatch_risk=self.estimate_support_mismatch_risk(),
        )

    def compute_coverage_score(self) -> float:
        """Fraction of occupied cells in a discretized state space (capped at 5D)."""
        n = min(len(self.obs), 50_000)
        sample = self.obs[:n]
        bins = 10

        obs_min = sample.min(axis=0)
        obs_range = sample.max(axis=0) - obs_min + 1e-8
        sample_norm = (sample - obs_min) / obs_range

        n_dims = min(sample_norm.shape[1], 5)
        sample_norm = sample_norm[:, :n_dims]

        occupied = set()
        for row in sample_norm:
            cell = tuple(int(np.clip(v * bins, 0, bins - 1)) for v in row)
            occupied.add(cell)

        total = bins ** n_dims
        return min(1.0, len(occupied) / total)

    def compute_behavior_entropy(self) -> float:
        """Entropy of discretized action distribution."""
        n_bins = 10
        act_min = self.actions.min(axis=0)
        act_range = self.actions.max(axis=0) - act_min + 1e-8
        act_norm = (self.actions - act_min) / act_range
        indices = (act_norm * n_bins).astype(int).clip(0, n_bins - 1)
        keys = [tuple(row) for row in indices]
        counts = Counter(keys)
        total = len(keys)
        probs = np.array([c / total for c in counts.values()])
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def compute_reward_stats(self) -> dict:
        r = self.rewards
        return {
            "mean": float(r.mean()),
            "std": float(r.std()),
            "min": float(r.min()),
            "max": float(r.max()),
            "skewness": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)),
            "pct_positive": float((r > 0).mean()),
            "pct_negative": float((r < 0).mean()),
            "histogram": np.histogram(r, bins=50)[0].tolist(),
        }

    def compute_episode_stats(self) -> dict:
        episode_lengths = []
        for ep_id in np.unique(self.episode_ids):
            mask = self.episode_ids == ep_id
            episode_lengths.append(int(mask.sum()))
        ep_lengths = np.array(episode_lengths)
        terminal_rate = float(self.dones.mean())
        return {
            "mean_length": float(ep_lengths.mean()),
            "std_length": float(ep_lengths.std()),
            "min_length": int(ep_lengths.min()),
            "max_length": int(ep_lengths.max()),
            "n_episodes": len(episode_lengths),
            "terminal_rate": terminal_rate,
            "histogram": np.histogram(ep_lengths, bins=20)[0].tolist(),
        }

    def detect_ood_risk(self) -> dict:
        """Estimate OOD risk by measuring how far random actions are from dataset."""
        n_samples = min(10_000, len(self.actions))
        act_sample = self.actions[:n_samples]
        random_actions = np.random.uniform(
            self.actions.min(axis=0),
            self.actions.max(axis=0),
            size=(n_samples, self.actions.shape[1]),
        )
        knn = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
        knn.fit(act_sample)
        dists, _ = knn.kneighbors(random_actions)
        mean_dist = float(dists.mean())
        act_range = float((self.actions.max(axis=0) - self.actions.min(axis=0)).mean())
        sparse_frac = float((dists[:, 0] > act_range * 0.3).mean())
        if sparse_frac < 0.2:
            risk_level = "low"
        elif sparse_frac < 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"
        return {
            "risk_level": risk_level,
            "sparse_state_fraction": sparse_frac,
            "mean_random_nn_dist": mean_dist,
            "explanation": (
                f"{sparse_frac*100:.1f}% of random actions are far from dataset support"
            ),
        }

    def detect_outlier_transitions(self) -> list:
        """IQR-based outlier detection on rewards."""
        r = self.rewards
        q1, q3 = np.percentile(r, 25), np.percentile(r, 75)
        iqr = q3 - q1
        outlier_mask = (r < q1 - 3 * iqr) | (r > q3 + 3 * iqr)
        return np.where(outlier_mask)[0].tolist()[:100]

    def compute_action_coverage(self) -> dict:
        result = {}
        a_min_global = self.actions.min(axis=0)
        a_max_global = self.actions.max(axis=0)
        for dim in range(self.actions.shape[1]):
            a = self.actions[:, dim]
            dim_range = float(a_max_global[dim] - a_min_global[dim] + 1e-8)
            coverage = float((a.max() - a.min()) / dim_range)
            result[f"dim_{dim}"] = {
                "min": float(a.min()),
                "max": float(a.max()),
                "mean": float(a.mean()),
                "std": float(a.std()),
                "coverage": min(1.0, coverage),
                "histogram": np.histogram(a, bins=20)[0].tolist(),
            }
        overall = float(
            np.mean([v["coverage"] for k, v in result.items() if isinstance(v, dict)])
        )
        result["overall_coverage"] = overall
        return result

    def compute_state_coverage_pca(self) -> dict:
        n = min(len(self.obs), 5_000)
        sample = self.obs[:n]
        n_components = min(2, sample.shape[1], sample.shape[0])
        pca = PCA(n_components=n_components)
        points = pca.fit_transform(sample)
        # Pad to 2 columns if needed
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros((points.shape[0], 2 - points.shape[1]))])
        return {
            "state_pca_points": points,
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        }

    def estimate_support_mismatch_risk(self) -> float:
        """k-NN density ratio between in-distribution and random state-actions."""
        n = min(5_000, len(self.obs))
        obs_sample = self.obs[:n]
        act_sample = self.actions[:n]
        sa = np.concatenate([obs_sample, act_sample], axis=1)
        knn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree")
        knn.fit(sa)
        dists, _ = knn.kneighbors(sa)
        mean_in_dist = float(dists[:, 1:].mean())

        sa_random = np.random.uniform(sa.min(axis=0), sa.max(axis=0), size=(n, sa.shape[1]))
        dists_rand, _ = knn.kneighbors(sa_random)
        mean_out_dist = float(dists_rand.mean())

        risk = float(np.clip((mean_out_dist / (mean_in_dist + 1e-8) - 1) / 10, 0.0, 1.0))
        return risk
