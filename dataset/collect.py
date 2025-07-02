import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from env.neuroquant_env import NeuroQuantEnv
from tqdm import tqdm


def scripted_policy(obs, agent_pos, agent_dir, goal_pos):
    """
    Greedy navigation to goal. Turns agent until it's facing goal, then moves forward.
    - agent_pos: (row, col)
    - agent_dir: 0=up, 1=right, 2=down, 3=left
    - goal_pos: (row, col)
    """
    agent_r, agent_c = agent_pos
    goal_r, goal_c = goal_pos

    # Compute desired direction
    if goal_r < agent_r:
        desired_dir = 0  # up
    elif goal_r > agent_r:
        desired_dir = 2  # down
    elif goal_c > agent_c:
        desired_dir = 1  # right
    elif goal_c < agent_c:
        desired_dir = 3  # left
    else:
        return 1  # Already at goal, default to forward

    # Compute turn needed
    if desired_dir == agent_dir:
        return 1  # move forward
    elif (desired_dir - agent_dir) % 4 == 1:
        return 2  # turn right
    else:
        return 0  # turn left



def collect_data(num_episodes=100):
    env = NeuroQuantEnv()
    env.obs_mode = "vector"  # ensure vector observations

    buffer = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "dones": []
    }

    episode_rewards = []
    episode_lengths = []

    for ep in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0
        max_steps = 500  # safety cap

        while not done and ep_len < max_steps:
            action = scripted_policy(obs, env.agent_pos, env.agent_dir, [9, 9])


            next_obs, reward, done, _, _ = env.step(action)

            buffer["observations"].append(obs)
            buffer["actions"].append(action)
            buffer["rewards"].append(reward)
            buffer["next_observations"].append(next_obs)
            buffer["dones"].append(done)

            obs = next_obs
            ep_reward += reward
            ep_len += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    env.close()

    # === Save dataset ===
    os.makedirs("dataset", exist_ok=True)
    np.savez_compressed(
        "dataset/replay_buffer.npz",
        observations=np.array(buffer["observations"]),
        actions=np.array(buffer["actions"]),
        rewards=np.array(buffer["rewards"]),
        next_observations=np.array(buffer["next_observations"]),
        dones=np.array(buffer["dones"])
    )

    # === Save reward histogram ===
    plt.hist(episode_rewards, bins=20)
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.title("Reward Distribution")
    plt.savefig("dataset/reward_histogram.png")

    # === Save metadata ===
    with open("dataset/metadata.txt", "w") as f:
        f.write(f"Num Episodes: {num_episodes}\n")
        f.write(f"Avg Episode Reward: {np.mean(episode_rewards):.2f}\n")
        f.write(f"Avg Episode Length: {np.mean(episode_lengths):.2f}\n")
        f.write(f"Total Transitions: {len(buffer['actions'])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    collect_data(num_episodes=args.episodes)
