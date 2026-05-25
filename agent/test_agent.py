# agent/test_agent.py
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.neuroquant_env import NeuroQuantEnv, render_episode_gif

def evaluate_policy(policy_fn, episodes=10, render=False, save_gif=False):
    env = NeuroQuantEnv(obs_mode="vector")  # 4D vector state
    rewards = []

    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            a = policy_fn(obs)
            obs, r, done, _, _ = env.step(a)
            total_reward += r
            if render:
                env.render()

        rewards.append(total_reward)
        print(f"[Episode {i}] Reward: {total_reward:.2f}")

    avg_reward = np.mean(rewards)
    print(f"\n✅ Average Reward over {episodes} episodes: {avg_reward:.2f}")

    if save_gif:
        os.makedirs("docs/replays", exist_ok=True)
        render_episode_gif(env, policy_fn, filename="docs/replays/test_run.gif", max_steps=100)
        print("🎥 Saved replay to docs/replays/test_run.gif")

    return avg_reward
