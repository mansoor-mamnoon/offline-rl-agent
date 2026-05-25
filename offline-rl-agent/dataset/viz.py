import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse

buffer = np.load("dataset/replay_buffer.npz")
observations = buffer["observations"]
actions = buffer["actions"]
rewards = buffer["rewards"]

os.makedirs("docs/plots", exist_ok=True)


def plot_tsne():
    sampled_obs = observations[::10]  # sample every 10th to speed up t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(sampled_obs)

    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=10, c="blue")
    plt.title("t-SNE of Observations")
    plt.savefig("docs/plots/tsne_obs.png")
    plt.close()



def plot_action_distribution():
    plt.figure()
    plt.hist(actions, bins=np.arange(actions.max()+2)-0.5, rwidth=0.8)
    plt.xticks(range(actions.max()+1))
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Action Distribution")
    plt.savefig("docs/plots/action_distribution.png")
    plt.close()


def plot_episode_rewards():
    episode_rewards = []
    running_total = 0

    for r, done in zip(rewards, buffer["dones"]):
        running_total += r
        if done:
            episode_rewards.append(running_total)
            running_total = 0

    plt.figure()
    plt.hist(episode_rewards, bins=20)
    plt.xlabel("Total Reward per Episode")
    plt.ylabel("Frequency")
    plt.title("Episode Reward Distribution")
    plt.savefig("docs/plots/episode_rewards.png")
    plt.close()



if __name__ == "__main__":
    plot_tsne()
    plot_action_distribution()
    plot_episode_rewards()
    print("Saved all visualizations in docs/plots/")

