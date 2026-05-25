import numpy as np
import h5py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GridWorldConfig:
    grid_size: int = 10
    start: tuple = (0, 0)
    goal: tuple = (9, 9)
    cliff_cells: list = field(default_factory=list)
    obstacle_cells: list = field(default_factory=list)
    max_steps: int = 200
    state_repr: str = "normalized"  # "normalized" or "onehot"


class GridWorld:
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

    def __init__(self, config: Optional[GridWorldConfig] = None):
        if config is None:
            config = GridWorldConfig()
        self.config = config
        self.grid_size = config.grid_size
        self.start = tuple(config.start)
        self.goal = tuple(config.goal)
        self.cliff_cells = set(map(tuple, config.cliff_cells))
        self.obstacle_cells = set(map(tuple, config.obstacle_cells))
        self.max_steps = config.max_steps
        self.state_repr = config.state_repr

        if self.state_repr == "onehot":
            self.obs_dim = self.grid_size * self.grid_size
        else:
            self.obs_dim = 2

        self.n_actions = 4
        self.agent_pos = self.start
        self.steps = 0
        self.done = False

    def reset(self) -> np.ndarray:
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> tuple:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # Clamp to grid boundaries
        nr = max(0, min(self.grid_size - 1, nr))
        nc = max(0, min(self.grid_size - 1, nc))

        # If obstacle, stay put
        if (nr, nc) in self.obstacle_cells:
            nr, nc = r, c

        self.agent_pos = (nr, nc)
        self.steps += 1

        # Compute reward
        if self.agent_pos == self.goal:
            reward = 10.0
            self.done = True
        elif self.agent_pos in self.cliff_cells:
            reward = -10.0
            self.done = True
        else:
            reward = -0.1

        if self.steps >= self.max_steps:
            self.done = True

        info = {
            "pos": self.agent_pos,
            "steps": self.steps,
            "reached_goal": self.agent_pos == self.goal,
            "fell_off_cliff": self.agent_pos in self.cliff_cells,
        }

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> np.ndarray:
        r, c = self.agent_pos
        if self.state_repr == "onehot":
            state = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
            state[r * self.grid_size + c] = 1.0
            return state
        else:
            return np.array([r / self.grid_size, c / self.grid_size], dtype=np.float32)

    def render(self) -> np.ndarray:
        cell_size = 20
        img = np.ones((self.grid_size * cell_size, self.grid_size * cell_size, 3), dtype=np.uint8) * 220

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rs, re = r * cell_size, (r + 1) * cell_size
                cs, ce = c * cell_size, (c + 1) * cell_size
                pos = (r, c)
                if pos == self.start:
                    img[rs:re, cs:ce] = [0, 200, 0]  # Green
                elif pos == self.goal:
                    img[rs:re, cs:ce] = [200, 0, 0]  # Red
                elif pos in self.obstacle_cells:
                    img[rs:re, cs:ce] = [30, 30, 30]  # Black
                elif pos in self.cliff_cells:
                    img[rs:re, cs:ce] = [200, 200, 0]  # Yellow

        # Draw agent
        ar, ac = self.agent_pos
        rs, re = ar * cell_size, (ar + 1) * cell_size
        cs, ce = ac * cell_size, (ac + 1) * cell_size
        img[rs:re, cs:ce] = [0, 0, 200]  # Blue

        return img

    def generate_dataset(self, policy, n_episodes: int = 1000, save_path: Optional[str] = None) -> dict:
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        episode_ids = []

        for ep in range(n_episodes):
            obs = self.reset()
            ep_done = False
            while not ep_done:
                action = policy(obs)
                next_obs, reward, ep_done, _ = self.step(action)
                observations.append(obs.copy())
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs.copy())
                dones.append(ep_done)
                episode_ids.append(ep)
                obs = next_obs

        dataset = {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
            "episode_ids": np.array(episode_ids, dtype=np.int64),
        }

        if save_path is not None:
            with h5py.File(save_path, "w") as f:
                for key, val in dataset.items():
                    f.create_dataset(key, data=val)

        return dataset


def random_policy(state: np.ndarray) -> int:
    return np.random.randint(4)


def near_optimal_policy(state: np.ndarray, goal: tuple = (9, 9), grid_size: int = 10) -> int:
    # Reconstruct position from state
    r = int(round(state[0] * grid_size))
    c = int(round(state[1] * grid_size))
    gr, gc = goal

    # Greedy Manhattan: prefer moving toward goal
    options = []
    dr = gr - r
    dc = gc - c

    if dr > 0:
        options.append(1)  # down
    elif dr < 0:
        options.append(0)  # up

    if dc > 0:
        options.append(3)  # right
    elif dc < 0:
        options.append(2)  # left

    if not options:
        options = [0, 1, 2, 3]

    return np.random.choice(options)


def mixed_policy(state: np.ndarray, optimal_prob: float = 0.7) -> int:
    if np.random.random() < optimal_prob:
        return near_optimal_policy(state)
    return random_policy(state)
