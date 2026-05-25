"""Traffic routing simulator with SLO constraints.

State: 32-dimensional vector describing 3 backend services.
Action: 4-dimensional continuous [0,1], route fractions (3) + shed fraction (1).
"""

import numpy as np
import h5py
from typing import Optional


# SLO thresholds
LATENCY_THRESHOLD_MS = 200.0
ERROR_THRESHOLD = 0.01
LOAD_THRESHOLD = 0.9


class TrafficRoutingEnv:
    """Traffic routing environment with 3 backends and SLO constraints."""

    N_BACKENDS = 3
    STATE_DIM = 32
    ACT_DIM = 4

    def __init__(
        self,
        max_steps: int = 200,
        latency_threshold: float = LATENCY_THRESHOLD_MS,
        error_threshold: float = ERROR_THRESHOLD,
        load_threshold: float = LOAD_THRESHOLD,
        stress_test: bool = False,
        seed: Optional[int] = None,
    ):
        self.max_steps = max_steps
        self.latency_threshold = latency_threshold
        self.error_threshold = error_threshold
        self.load_threshold = load_threshold
        self.stress_test = stress_test

        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.done = False

        # Incident state
        self._incident_active = np.zeros(self.N_BACKENDS, dtype=bool)
        self._incident_duration = np.zeros(self.N_BACKENDS, dtype=int)

        # Internal state variables (pre-noise)
        self._service_load = np.full(self.N_BACKENDS, 0.3, dtype=np.float32)
        self._request_rate = np.full(self.N_BACKENDS, 0.3, dtype=np.float32)
        self._p95_latency = np.full(self.N_BACKENDS, 50.0, dtype=np.float32)  # ms
        self._error_rate = np.zeros(self.N_BACKENDS, dtype=np.float32)
        self._region_health = np.ones(self.N_BACKENDS, dtype=np.float32)
        self._queue_depth = np.zeros(self.N_BACKENDS, dtype=np.float32)
        self._prior_route_alloc = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float32)
        self._cpu_usage = np.full(self.N_BACKENDS, 0.3, dtype=np.float32)

        self._global_request_rate = 0.5
        self._rolling_slo_violation_rate = 0.0
        self._memory_usage_global = 0.4

        # For SLO tracking
        self._slo_window = []

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.done = False
        self._incident_active = np.zeros(self.N_BACKENDS, dtype=bool)
        self._incident_duration = np.zeros(self.N_BACKENDS, dtype=int)
        self._service_load = np.full(self.N_BACKENDS, 0.3, dtype=np.float32)
        self._request_rate = np.full(self.N_BACKENDS, 0.3, dtype=np.float32)
        self._p95_latency = np.full(self.N_BACKENDS, 50.0, dtype=np.float32)
        self._error_rate = np.zeros(self.N_BACKENDS, dtype=np.float32)
        self._region_health = np.ones(self.N_BACKENDS, dtype=np.float32)
        self._queue_depth = np.zeros(self.N_BACKENDS, dtype=np.float32)
        self._prior_route_alloc = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float32)
        self._cpu_usage = np.full(self.N_BACKENDS, 0.3, dtype=np.float32)
        self._global_request_rate = 0.5
        self._rolling_slo_violation_rate = 0.0
        self._memory_usage_global = 0.4
        self._slo_window = []
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple:
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        action = np.asarray(action, dtype=np.float32).copy()

        # Normalize routing fractions (first 3 dims)
        route_fracs = action[:3]
        shed_frac = float(np.clip(action[3], 0.0, 1.0))
        route_fracs = np.clip(route_fracs, 0.0, 1.0)
        route_sum = route_fracs.sum()
        if route_sum < 1e-8:
            route_fracs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float32)
        else:
            route_fracs = route_fracs / route_sum

        self.step_count += 1

        # Update time-of-day signal
        hour = (self.step_count / self.max_steps) * 24.0
        self._time_of_day_sin = np.sin(2 * np.pi * hour / 24.0)
        self._time_of_day_cos = np.cos(2 * np.pi * hour / 24.0)

        # Update global request rate with diurnal pattern
        base_rate = 0.5 + 0.3 * (
            np.sin(2 * np.pi * (hour - 9) / 24.0) + np.sin(2 * np.pi * (hour - 18) / 24.0)
        ) / 2.0
        base_rate = float(np.clip(base_rate, 0.1, 1.0))

        # Stress test: inject 3x spike at midpoint
        if self.stress_test and self.step_count == self.max_steps // 2:
            base_rate = min(1.0, base_rate * 3.0)

        self._global_request_rate = float(base_rate + self.rng.normal(0, 0.01))
        self._global_request_rate = float(np.clip(self._global_request_rate, 0.0, 1.0))

        # Update incident states
        for b in range(self.N_BACKENDS):
            if self._incident_active[b]:
                self._incident_duration[b] += 1
                # Geometric duration p=0.1: end with prob 0.1 each step
                if self.rng.random() < 0.1:
                    self._incident_active[b] = False
                    self._incident_duration[b] = 0
            else:
                # Poisson(0.02) arrival
                if self.rng.random() < 0.02:
                    self._incident_active[b] = True
                    self._incident_duration[b] = 1

        # Update region health
        self._region_health = np.where(self._incident_active, 0.0, 1.0).astype(np.float32)

        # Update load: load = prior_load * decay + route_frac * global_rate
        decay = 0.8
        incoming = route_fracs * self._global_request_rate * (1.0 - shed_frac)
        self._service_load = (decay * self._service_load + (1 - decay) * incoming).astype(np.float32)
        self._service_load = np.clip(self._service_load, 0.0, 1.0)

        # Update request rate
        self._request_rate = (0.9 * self._request_rate + 0.1 * route_fracs * self._global_request_rate).astype(np.float32)

        # Update queue depth
        self._queue_depth = np.clip(
            self._service_load - 0.7, 0.0, None
        ).astype(np.float32)

        # CPU proportional to load
        self._cpu_usage = np.clip(
            self._service_load + self.rng.normal(0, 0.01, self.N_BACKENDS), 0.0, 1.0
        ).astype(np.float32)

        # Memory usage
        self._memory_usage_global = float(np.clip(
            0.4 + 0.3 * self._service_load.mean() + self.rng.normal(0, 0.01), 0.0, 1.0
        ))

        # Compute latency: sigmoid spike above 0.7 load
        for b in range(self.N_BACKENDS):
            load = float(self._service_load[b])
            base_latency = 100.0 / (1.0 + np.exp(-10.0 * (load - 0.7)))
            incident_bump = 200.0 if self._incident_active[b] else 0.0
            noise = float(self.rng.normal(0, 5.0))
            self._p95_latency[b] = float(np.clip(base_latency + incident_bump + noise, 0.0, 2000.0))

        # Compute error rate: spikes when load > 0.85 or incident
        for b in range(self.N_BACKENDS):
            base_err = 0.001
            if self._service_load[b] > 0.85:
                base_err += 0.05 * (self._service_load[b] - 0.85) / 0.15
            if self._incident_active[b]:
                base_err += 0.05
            noise = float(self.rng.normal(0, 0.002))
            self._error_rate[b] = float(np.clip(base_err + noise, 0.0, 1.0))

        # SLO violation check
        slo_violated = bool(
            any(self._p95_latency > self.latency_threshold)
            or any(self._error_rate > self.error_threshold)
            or any(self._service_load > self.load_threshold)
        )
        self._slo_window.append(slo_violated)
        if len(self._slo_window) > 50:
            self._slo_window.pop(0)
        self._rolling_slo_violation_rate = float(np.mean(self._slo_window))

        # Compute reward components
        requests_served_fraction = 1.0 - shed_frac
        avg_latency = float(self._p95_latency.mean())
        avg_error = float(self._error_rate.mean())
        infrastructure_cost = float(self._service_load.mean())

        reward = (
            1.0 * requests_served_fraction
            - 2.0 * max(0.0, avg_latency - self.latency_threshold) / self.latency_threshold
            - 3.0 * avg_error
            - 0.5 * infrastructure_cost
            - 5.0 * float(slo_violated)
            - 1.0 * shed_frac
        )

        # Update prior allocation
        self._prior_route_alloc = route_fracs.copy()

        if self.step_count >= self.max_steps:
            self.done = True

        info = {
            "slo_violated": slo_violated,
            "p95_latency": self._p95_latency.copy(),
            "error_rate": self._error_rate.copy(),
            "service_load": self._service_load.copy(),
            "incident_active": self._incident_active.copy(),
            "shed_fraction": shed_frac,
            "requests_served_fraction": requests_served_fraction,
        }

        return self._get_state(), float(reward), self.done, info

    def _get_state(self) -> np.ndarray:
        """Build 32-dimensional state vector."""
        # time_of_day init if not set
        if not hasattr(self, "_time_of_day_sin"):
            self._time_of_day_sin = 0.0
            self._time_of_day_cos = 1.0

        state = np.concatenate([
            self._service_load,                                          # 3
            self._request_rate,                                          # 3
            self._p95_latency / 1000.0,                                  # 3 (ms/1000)
            self._error_rate,                                            # 3
            self._region_health,                                         # 3
            self._queue_depth,                                           # 3
            self._prior_route_alloc,                                     # 3
            [self._global_request_rate],                                 # 1
            [self._time_of_day_sin, self._time_of_day_cos],             # 2
            self._incident_active.astype(np.float32),                   # 3
            [self._rolling_slo_violation_rate],                         # 1
            self._cpu_usage,                                             # 3
            [self._memory_usage_global],                                 # 1
        ]).astype(np.float32)

        assert state.shape == (32,), f"State shape is {state.shape}, expected (32,)"
        # Add small Gaussian noise
        state = state + self.rng.normal(0, 0.01, 32).astype(np.float32)
        # Clip to reasonable range
        state = np.clip(state, -2.0, 2.0).astype(np.float32)
        return state

    def generate_dataset(
        self,
        policy_fn,
        n_episodes: int = 1000,
        max_steps: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> dict:
        if max_steps is not None:
            orig_max = self.max_steps
            self.max_steps = max_steps

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        episode_ids = []

        episode_metadata = []

        for ep in range(n_episodes):
            obs = self.reset()
            ep_done = False
            ep_reward = 0.0
            ep_slo_violations = 0
            ep_incident_count = 0

            while not ep_done:
                action = policy_fn(obs)
                next_obs, reward, ep_done, info = self.step(action)

                observations.append(obs.copy())
                actions.append(np.asarray(action, dtype=np.float32).copy())
                rewards.append(reward)
                next_observations.append(next_obs.copy())
                dones.append(ep_done)
                episode_ids.append(ep)

                ep_reward += reward
                ep_slo_violations += int(info["slo_violated"])
                ep_incident_count += int(any(info["incident_active"]))

                obs = next_obs

            episode_metadata.append({
                "episode_id": ep,
                "total_reward": ep_reward,
                "slo_violations": ep_slo_violations,
                "incident_count": ep_incident_count,
            })

        if max_steps is not None:
            self.max_steps = orig_max

        dataset = {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
            "episode_ids": np.array(episode_ids, dtype=np.int64),
        }

        if save_path is not None:
            with h5py.File(save_path, "w") as f:
                for key, val in dataset.items():
                    f.create_dataset(key, data=val)
                # Save metadata
                meta_rewards = np.array([m["total_reward"] for m in episode_metadata])
                meta_slo = np.array([m["slo_violations"] for m in episode_metadata])
                meta_inc = np.array([m["incident_count"] for m in episode_metadata])
                f.create_dataset("meta_episode_rewards", data=meta_rewards)
                f.create_dataset("meta_slo_violations", data=meta_slo)
                f.create_dataset("meta_incident_counts", data=meta_inc)

        return dataset


# ─── Behavior policies ──────────────────────────────────────────────────────


def random_policy(state: np.ndarray) -> np.ndarray:
    """Uniform random routing."""
    action = np.random.uniform(0, 1, 4).astype(np.float32)
    # Normalize routing fracs
    action[:3] = action[:3] / action[:3].sum()
    return action


_round_robin_counter = [0]


def round_robin_policy(state: np.ndarray) -> np.ndarray:
    """Cycle routing: [1,0,0] -> [0,1,0] -> [0,0,1] -> ..."""
    idx = _round_robin_counter[0] % 3
    action = np.zeros(4, dtype=np.float32)
    action[idx] = 1.0
    action[3] = 0.0
    _round_robin_counter[0] += 1
    return action


def load_aware_policy(state: np.ndarray) -> np.ndarray:
    """Route inversely proportional to load via softmax(-load)."""
    service_load = state[:3].copy()
    # Softmax of negative load
    neg_load = -service_load
    neg_load -= neg_load.max()
    exp_load = np.exp(neg_load)
    fracs = exp_load / exp_load.sum()
    action = np.zeros(4, dtype=np.float32)
    action[:3] = fracs
    action[3] = 0.0
    return action


def suboptimal_policy(state: np.ndarray) -> np.ndarray:
    """Load-aware but ignores incident flags (columns 27-29 of state)."""
    # Same as load_aware but doesn't adjust for incidents
    service_load = state[:3].copy()
    neg_load = -service_load
    neg_load -= neg_load.max()
    exp_load = np.exp(neg_load)
    fracs = exp_load / exp_load.sum()
    # Sometimes shed a little (suboptimal)
    action = np.zeros(4, dtype=np.float32)
    action[:3] = fracs
    action[3] = 0.05
    return action


POLICY_MAP = {
    "random": random_policy,
    "round_robin": round_robin_policy,
    "load_aware": load_aware_policy,
    "suboptimal": suboptimal_policy,
}


def get_policy(name: str):
    if name not in POLICY_MAP:
        raise ValueError(f"Unknown policy '{name}'. Choose from {list(POLICY_MAP.keys())}")
    return POLICY_MAP[name]
