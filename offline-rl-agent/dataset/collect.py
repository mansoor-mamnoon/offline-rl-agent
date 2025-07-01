import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from env.neuroquant_env import NeuroQuantEnv
from tqdm import tqdm




def collect_data(num_episodes=100):
    env = NeuroQuantEnv()
    env.obs_mode = "vector"

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

        while not done:
            action = env.action_space.sample()
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


    np.savez_compressed(
    "dataset/replay_buffer.npz",
    observations=np.array(buffer["observations"]),
    actions=np.array(buffer["actions"]),
    rewards=np.array(buffer["rewards"]),
    next_observations=np.array(buffer["next_observations"]),
    dones=np.array(buffer["dones"])
)
    plt.hist(episode_rewards, bins=20)
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.title("Reward Distribution")
    plt.savefig("dataset/reward_histogram.png")

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



