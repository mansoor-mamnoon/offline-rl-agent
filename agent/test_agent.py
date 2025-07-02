# agent/test_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from env.neuroquant_env import NeuroQuantEnv, render_episode_gif
from cql import CQLAgent

# === 1. Setup ===
env = NeuroQuantEnv(obs_mode="vector")  # ensure you're using the 4D vector mode
obs_dim = 4
action_dim = 3

agent = CQLAgent(state_dim=obs_dim, action_dim=action_dim)
agent.q_net.load_state_dict(torch.load("checkpoints/best_q.pt"))
agent.policy.load_state_dict(torch.load("checkpoints/best_policy.pt"))

# Set to eval mode
agent.q_net.eval()
agent.q_target.eval()
agent.policy.eval()

def policy_fn(obs):
    """Convert obs to tensor, run policy, return int action."""
    obs_np = np.array(obs, dtype=np.float32)  # clean shape (4,)
    obs_tensor = torch.tensor(obs_np).unsqueeze(0)  # shape (1, 4)
    with torch.no_grad():
        logits = agent.policy(obs_tensor)
        return torch.argmax(logits, dim=1).item()

# === 2. Evaluate Agent ===
rewards = []
episodes = 10

for i in range(episodes):
    obs, _ = env.reset()  # FIXED: unpack properly
    done = False
    total_reward = 0

    while not done:
        a = policy_fn(obs)
        obs, r, done, _, _ = env.step(a)  # use correct unpacking
        total_reward += r
        env.render()

    rewards.append(total_reward)
    print(f"[Episode {i}] Reward: {total_reward:.2f}")

avg_reward = np.mean(rewards)
print(f"\n✅ Average Reward over {episodes} episodes: {avg_reward:.2f}")

# === 3. Save Replay ===
render_episode_gif(env, policy_fn, filename="docs/replays/test_run.gif", max_steps=100)
print("🎥 Saved replay to docs/replays/test_run.gif")
