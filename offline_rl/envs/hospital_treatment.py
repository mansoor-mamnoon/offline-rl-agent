"""Hospital Treatment Simulator for offline RL.

A simplified clinical decision support environment where an agent recommends
treatment intensities for patients based on their health state. Designed for
offline RL research with safety constraints (avoid harmful over/under-treatment).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import h5py
from pathlib import Path


@dataclass
class HospitalConfig:
    n_features: int = 16  # patient state features
    max_steps: int = 20  # treatment episodes (days)
    # Thresholds
    critical_threshold: float = 0.8  # health score below this = critical
    recovery_threshold: float = 0.3  # health score below this = recovered
    # Reward weights
    recovery_reward: float = 10.0
    death_penalty: float = -10.0
    treatment_cost: float = 0.5  # per unit of treatment
    side_effect_penalty: float = 2.0  # for over-treatment


class HospitalTreatmentEnv:
    """
    Patient health management environment.

    State (16-dim):
      [0]   disease_severity       [0,1]
      [1]   immune_response        [0,1]
      [2]   inflammation_marker    [0,1]
      [3]   vital_signs_score      [0,1]
      [4]   medication_level       [0,1]  (current drug in system)
      [5]   age_normalized         [0,1]
      [6]   comorbidity_score      [0,1]
      [7]   days_in_treatment      [0,1]  (normalized by max_steps)
      [8:12] lab_values            [0,1] x4
      [12]  pain_score             [0,1]
      [13]  mobility_score         [0,1]
      [14]  recent_side_effects    [0,1]  (rolling 3-day)
      [15]  time_since_admission   [0,1]

    Action (2-dim continuous [0,1]):
      [0] drug_dose           (treatment intensity)
      [1] supportive_care     (supplemental care level)

    Reward:
      +10   if patient recovers (severity < 0.2)
      -10   if patient dies (severity > 0.95 or steps exhausted with severity > 0.5)
      -0.5  per unit drug dose (cost)
      -2.0  if drug_dose > 0.8 and medication_level > 0.6 (over-treatment)
      -0.1  per step (urgency)
    """

    def __init__(self, config: Optional[HospitalConfig] = None, seed: int = None):
        self.config = config or HospitalConfig()
        self.rng = np.random.default_rng(seed)
        self.state = None
        self._step = 0
        self._patient_profile = None

    def reset(self) -> np.ndarray:
        self._step = 0
        self._patient_profile = {
            "age": self.rng.uniform(0.2, 0.9),
            "comorbidity": self.rng.uniform(0.0, 0.7),
            "disease_aggressiveness": self.rng.uniform(0.3, 0.8),
        }
        self.state = self._init_state()
        return self.state.copy()

    def _init_state(self) -> np.ndarray:
        s = np.zeros(self.config.n_features)
        s[0] = self.rng.uniform(0.4, 0.85)  # disease severity (moderate-severe)
        s[1] = self.rng.uniform(0.2, 0.6)  # immune response
        s[2] = self.rng.uniform(0.3, 0.7)  # inflammation
        s[3] = self.rng.uniform(0.4, 0.8)  # vitals
        s[4] = 0.0  # medication_level (none yet)
        s[5] = self._patient_profile["age"]
        s[6] = self._patient_profile["comorbidity"]
        s[7] = 0.0  # days in treatment
        s[8:12] = self.rng.uniform(0.3, 0.7, 4)  # lab values
        s[12] = self.rng.uniform(0.3, 0.7)  # pain
        s[13] = self.rng.uniform(0.5, 0.9)  # mobility
        s[14] = 0.0  # side effects
        s[15] = 0.0  # time since admission
        return s

    def step(self, action: np.ndarray):
        action = np.clip(action, 0, 1)
        drug_dose, supportive_care = float(action[0]), float(action[1])

        # Update medication level (pharmacokinetics)
        self.state[4] = 0.85 * self.state[4] + 0.15 * drug_dose

        # Disease progression/regression
        treatment_effect = (
            drug_dose
            * 0.15
            * (1.0 - 0.3 * self._patient_profile["disease_aggressiveness"])
        )
        natural_progression = 0.02 * self._patient_profile["disease_aggressiveness"]
        noise = self.rng.normal(0, 0.02)

        self.state[0] = np.clip(
            self.state[0] - treatment_effect + natural_progression + noise, 0, 1
        )
        self.state[1] = np.clip(self.state[1] + 0.05 * drug_dose - 0.01, 0, 1)
        self.state[2] = np.clip(
            0.7 * self.state[2] + 0.3 * self.state[0] + self.rng.normal(0, 0.02), 0, 1
        )
        self.state[3] = np.clip(
            1.0 - 0.5 * self.state[0] + 0.1 * supportive_care + self.rng.normal(0, 0.02),
            0,
            1,
        )
        self.state[8:12] = np.clip(
            self.state[8:12] * 0.9 + 0.1 * self.state[0] + self.rng.normal(0, 0.01, 4),
            0,
            1,
        )
        self.state[12] = np.clip(
            0.8 * self.state[12] + 0.2 * self.state[0] - 0.05 * supportive_care, 0, 1
        )
        self.state[13] = np.clip(
            1.0 - 0.4 * self.state[0] + 0.05 * supportive_care + self.rng.normal(0, 0.02),
            0,
            1,
        )
        # Side effects from over-treatment
        over_treatment = max(0.0, drug_dose - 0.7) * max(0.0, self.state[4] - 0.5)
        self.state[14] = np.clip(0.7 * self.state[14] + 0.3 * over_treatment, 0, 1)
        # Time tracking
        self._step += 1
        self.state[7] = self._step / self.config.max_steps
        self.state[15] = self._step / self.config.max_steps

        # Compute reward
        reward = -0.1  # step penalty
        reward -= self.config.treatment_cost * drug_dose

        # Over-treatment penalty
        if drug_dose > 0.8 and self.state[4] > 0.6:
            reward -= self.config.side_effect_penalty * over_treatment

        # Check terminal conditions
        recovered = bool(self.state[0] < 0.2)
        died = bool(self.state[0] > 0.95)
        timed_out = self._step >= self.config.max_steps

        done = recovered or died or (timed_out and self.state[0] > 0.5)

        if recovered:
            reward += self.config.recovery_reward
        elif died or (timed_out and self.state[0] > 0.5):
            reward += self.config.death_penalty

        info = {
            "recovered": recovered,
            "died": died,
            "disease_severity": float(self.state[0]),
            "medication_level": float(self.state[4]),
            "side_effects": float(self.state[14]),
            "slo_violated": bool(self.state[14] > 0.5),
        }

        return self.state.copy(), float(reward), done, info

    def render(self) -> np.ndarray:
        """Return RGB array showing patient status dashboard."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(6, 4), dpi=80)
        fig.suptitle(f"Patient Status (Day {self._step})", fontsize=10, fontweight="bold")

        axes[0, 0].barh(
            ["Severity"],
            [self.state[0]],
            color="red" if self.state[0] > 0.7 else "orange",
        )
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_title("Disease Severity", fontsize=8)

        axes[0, 1].barh(["Drug Level"], [self.state[4]], color="steelblue")
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_title("Medication Level", fontsize=8)

        labels = ["Lab1", "Lab2", "Lab3", "Lab4", "Pain", "Mobility"]
        values = list(self.state[8:12]) + [self.state[12], self.state[13]]
        axes[1, 0].bar(labels, values, color="gray", alpha=0.7)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title("Lab Values", fontsize=8)
        axes[1, 0].tick_params(axis="x", labelsize=6)

        color = "red" if self.state[14] > 0.5 else "green"
        axes[1, 1].barh(["Side Effects"], [self.state[14]], color=color)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_title("Side Effects", fontsize=8)

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return buf

    def generate_dataset(
        self,
        policy_name: str = "moderate",
        n_episodes: int = 500,
        save_path: Optional[str] = None,
    ) -> dict:
        """Generate offline dataset using named behavior policy."""
        policies = {
            "conservative": lambda s: np.array([0.3, 0.5]),
            "aggressive": lambda s: np.array([0.8, 0.3]),
            "moderate": lambda s: np.array([np.clip(s[0] * 0.8, 0.2, 0.7), 0.4]),
            "random": lambda s: self.rng.uniform(0, 1, 2),
        }
        policy = policies.get(policy_name, policies["moderate"])

        observations, actions, rewards, next_observations, dones, episode_ids = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for ep in range(n_episodes):
            obs = self.reset()
            done = False
            while not done:
                action = policy(obs)
                next_obs, reward, done, _ = self.step(action)
                observations.append(obs.copy())
                actions.append(action.copy())
                rewards.append(reward)
                next_observations.append(next_obs.copy())
                dones.append(float(done))
                episode_ids.append(ep)
                obs = next_obs

        dataset = {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "episode_ids": np.array(episode_ids, dtype=np.int32),
        }

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(save_path, "w") as f:
                for k, v in dataset.items():
                    f.create_dataset(k, data=v, compression="gzip")

        return dataset
