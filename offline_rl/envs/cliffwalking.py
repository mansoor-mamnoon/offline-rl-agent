"""Cliff Walking environment (Sutton & Barto layout).

4x12 grid. Cliff cells along row 3 (bottom row), columns 1-10.
Start: (3, 0), Goal: (3, 11).
"""

from offline_rl.envs.gridworld import GridWorld, GridWorldConfig


def make_cliff_walking() -> GridWorld:
    """Create the Cliff Walking environment from Sutton & Barto."""
    # Cliff cells: row 3 (bottom), columns 1..10
    cliff_cells = [(3, c) for c in range(1, 11)]

    config = GridWorldConfig(
        grid_size=12,  # Use 12 as the column dimension; we'll override
        start=(3, 0),
        goal=(3, 11),
        cliff_cells=cliff_cells,
        obstacle_cells=[],
        max_steps=200,
        state_repr="normalized",
    )

    return CliffWalking(config, n_rows=4, n_cols=12)


class CliffWalking(GridWorld):
    """Cliff Walking: 4x12 grid, rectangular layout.

    Inherits from GridWorld but supports non-square grids.
    """

    def __init__(self, config: GridWorldConfig, n_rows: int = 4, n_cols: int = 12):
        self.n_rows = n_rows
        self.n_cols = n_cols
        # Override grid_size to use column count for boundary checks
        super().__init__(config)
        # Override obs_dim for normalized repr
        self.obs_dim = 2

    def _get_state(self):
        import numpy as np

        r, c = self.agent_pos
        return np.array([r / self.n_rows, c / self.n_cols], dtype="float32")

    def step(self, action: int):

        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        nr = max(0, min(self.n_rows - 1, r + dr))
        nc = max(0, min(self.n_cols - 1, c + dc))

        if (nr, nc) in self.obstacle_cells:
            nr, nc = r, c

        self.agent_pos = (nr, nc)
        self.steps += 1

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
