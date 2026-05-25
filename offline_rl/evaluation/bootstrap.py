"""Bootstrap confidence interval utilities."""

import numpy as np
from typing import Callable


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        estimates.append(float(statistic(sample)))
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(estimates, alpha * 100)),
        float(np.percentile(estimates, (1 - alpha) * 100)),
    )


def bootstrap_ci_mean(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    return bootstrap_ci(data, np.mean, n_bootstrap, ci)


def bootstrap_ci_cvar(
    data: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    ci: float = 0.95,
) -> tuple:
    """Bootstrap CI for CVaR at given alpha level."""

    def cvar(x):
        n = max(1, int(len(x) * alpha))
        return float(np.sort(x)[:n].mean())

    return bootstrap_ci(data, cvar, n_bootstrap, ci)


def standard_error(data: np.ndarray) -> float:
    """Standard error of the mean."""
    return float(np.std(data) / np.sqrt(max(len(data), 1)))
