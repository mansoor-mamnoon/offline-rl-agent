import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class NeuroQuantEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        # Grid + Game Setup
        self.grid_size = 10  # 10x10 grid
        self.window_size = 500  # pixels
        self.cell_size = self.window_size // self.grid_size

        # 0 = empty, 1 = wall, 2 = goal
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.grid[2, 2] = 1
        self.grid[3, 4] = 1
        self.grid[5, 6] = 1
        self.grid[9, 9] = 2  # Goal at bottom-right

        # Agent state
        self.agent_pos = np.array([0, 0])  # starts top-left
        self.agent_dir = 1  # 0=up, 1=right, 2=down, 3=left

        # Observation = 3x3 local grid
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(3, 3),
            dtype=np.int8
        )

        # Action: turn left, go forward, turn right
        self.action_space = spaces.Discrete(3)

        # PyGame setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None




    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent
        self.agent_pos = np.array([0, 0])
        self.agent_dir = 1  # start facing right

        observation = self._get_obs()
        return observation, {}


    def step(self, action):
        # Update direction
        if action == 0:  # left turn
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 2:  # right turn
            self.agent_dir = (self.agent_dir + 1) % 4

        # Compute forward move
        forward = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }[self.agent_dir]

        new_pos = self.agent_pos + np.array(forward)

        # Check bounds and wall
        if (
            0 <= new_pos[0] < self.grid_size and
            0 <= new_pos[1] < self.grid_size and
            self.grid[new_pos[0], new_pos[1]] != 1
        ):
            self.agent_pos = new_pos  # move allowed

        reward = -0.1
        terminated = False
        truncated = False

        if self.grid[self.agent_pos[0], self.agent_pos[1]] == 2:
            reward = 10
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}


    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("NeuroQuantEnv")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))  # white

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = (255, 255, 255)  # default empty

                if self.grid[r, c] == 1:
                    color = (0, 0, 0)  # wall = black
                elif self.grid[r, c] == 2:
                    color = (0, 255, 0)  # goal = green

                pygame.draw.rect(
                    self.window,
                    color,
                    pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                )

        # Draw agent
        x = self.agent_pos[1] * self.cell_size
        y = self.agent_pos[0] * self.cell_size
        pygame.draw.rect(
            self.window,
            (255, 0, 0),
            pygame.Rect(x, y, self.cell_size, self.cell_size)
        )

        # Draw direction indicator (line)
        center = (x + self.cell_size // 2, y + self.cell_size // 2)
        delta = {
            0: (0, -self.cell_size // 2),
            1: (self.cell_size // 2, 0),
            2: (0, self.cell_size // 2),
            3: (-self.cell_size // 2, 0),
        }[self.agent_dir]

        end = (center[0] + delta[0], center[1] + delta[1])
        pygame.draw.line(self.window, (0, 0, 255), center, end, 3)

        pygame.display.flip()
        self.clock.tick(10)  # cap to 10 FPS




    def close(self):
        if self.window is not None:
            pygame.quit()

    def _get_obs(self):
        row, col = self.agent_pos
        padded_grid = np.pad(self.grid, pad_width=1, mode='constant', constant_values=1)

        # Offset agent position for padded grid
        row += 1
        col += 1

        local_obs = padded_grid[row - 1:row + 2, col - 1:col + 2]
        return local_obs

