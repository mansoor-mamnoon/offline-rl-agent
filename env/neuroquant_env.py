import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class NeuroQuantEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        # Define observation and action space
        # Example: Discrete 3-action agent, 2D vector state
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # PyGame setup
        self.render_mode = render_mode
        self.window_size = 500
        self.window = None
        self.clock = None

        self.agent_pos = np.array([5, 5])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([5, 5])  # within a 10x10 grid
        observation = self.agent_pos.astype(np.float32)
        return observation, {}

    def step(self, action):
        # Dummy logic: Move agent around
        if action == 0:   # left
            self.agent_pos[0] -= 1
        elif action == 1: # right
            self.agent_pos[0] += 1
        elif action == 2: # up
            self.agent_pos[1] -= 1

        reward = -0.1
        terminated = False
        truncated = False

        observation = self.agent_pos.astype(np.float32)
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("NeuroQuantEnv")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))  # white background

        scale = self.window_size // 10  # Assume a 10x10 grid

        # Clamp agent position to window grid
        x = max(0, min(self.agent_pos[0], 9)) * scale
        y = max(0, min(self.agent_pos[1], 9)) * scale

        pygame.draw.rect(
            self.window,
            (255, 0, 0),  # Red color
            pygame.Rect(x, y, scale, scale)
        )

        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS



    def close(self):
        if self.window is not None:
            pygame.quit()
