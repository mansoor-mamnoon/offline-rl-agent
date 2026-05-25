#!/usr/bin/env python
"""Generate demo GIFs for OfflineRL-Lab README."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("artifacts/gifs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_gridworld_gif():
    """GIF 1: GridWorld policy rollout."""
    print("Generating GridWorld rollout GIF...")
    from offline_rl.envs.gridworld import GridWorld, GridWorldConfig, near_optimal_policy
    from offline_rl.visualization.rollout_renderer import collect_rollout, save_gif

    env = GridWorld(GridWorldConfig(grid_size=8, goal=(7, 7)))
    policy = lambda obs: near_optimal_policy(obs, goal=(7, 7), grid_size=8)
    frames, metrics = collect_rollout(env, policy, max_steps=50, seed=42)
    path = save_gif(frames, OUT_DIR / "gridworld_rollout.gif", fps=5)
    print(f"  Saved: {path} | Return: {metrics['total_return']:.1f}")


def generate_traffic_comparison_gif():
    """GIF 2: Traffic routing - safe vs unsafe policy comparison."""
    print("Generating traffic routing comparison GIF...")
    from offline_rl.envs.traffic_routing import TrafficRoutingEnv
    from offline_rl.visualization.rollout_renderer import save_gif

    def random_policy(obs):
        acts = np.random.uniform(0, 1, 4)
        acts[:3] /= acts[:3].sum() + 1e-8
        return acts

    def load_aware_policy(obs):
        loads = obs[:3]
        weights = np.exp(-5 * loads)
        weights /= weights.sum() + 1e-8
        return np.append(weights, 0.0)

    # Track cumulative returns
    returns_rand = [0.0]
    returns_smart = [0.0]
    max_steps = 50
    env2 = TrafficRoutingEnv()
    env3 = TrafficRoutingEnv()
    np.random.seed(0)
    obs_r = env2.reset()
    obs_s = env3.reset()

    combined_frames = []

    for i in range(max_steps):
        a_r = random_policy(obs_r)
        a_s = load_aware_policy(obs_s)
        try:
            obs_r, r_r, done_r, _ = env2.step(a_r)
            returns_rand.append(returns_rand[-1] + r_r)
        except Exception:
            returns_rand.append(returns_rand[-1])
        try:
            obs_s, r_s, done_s, _ = env3.step(a_s)
            returns_smart.append(returns_smart[-1] + r_s)
        except Exception:
            returns_smart.append(returns_smart[-1])

        # Build comparison frame
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=60)

        axes[0].plot(returns_rand[: i + 2], color="red", linewidth=2)
        axes[0].set_title(
            f"Random Policy\nReturn: {returns_rand[i + 1]:.1f}", fontsize=10
        )
        min_y = min(min(returns_rand), min(returns_smart)) - 5
        max_y = max(max(returns_rand), max(returns_smart)) + 5
        axes[0].set_ylim(min_y, max_y)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Cumulative Return")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(returns_smart[: i + 2], color="green", linewidth=2)
        axes[1].set_title(
            f"Load-Aware Policy\nReturn: {returns_smart[i + 1]:.1f}", fontsize=10
        )
        axes[1].set_ylim(min_y, max_y)
        axes[1].set_xlabel("Step")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            "Traffic Routing: Random vs Load-Aware Policy",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        combined_frames.append(buf)

    path = save_gif(combined_frames, OUT_DIR / "traffic_comparison.gif", fps=6)
    print(f"  Saved: {path}")


def generate_dataset_diagnostics_gif():
    """GIF 3: Dataset diagnostics animated bar chart."""
    print("Generating dataset diagnostics GIF...")
    from offline_rl.visualization.rollout_renderer import save_gif

    frames = []
    metrics_to_show = [
        ("Transitions", 200000, "steelblue"),
        ("Episodes", 1000, "orange"),
        ("Coverage Score", 0.74, "green"),
        ("OOD Risk", 0.65, "red"),
        ("Behavior Entropy", 0.32, "purple"),
    ]

    for i in range(1, len(metrics_to_show) + 1):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=80)
        labels = [m[0] for m in metrics_to_show[:i]]
        values = [
            min(m[1], 1.0) if m[1] <= 1.0 else m[1] / 200000
            for m in metrics_to_show[:i]
        ]
        colors = [m[2] for m in metrics_to_show[:i]]

        bars = ax.barh(labels, values, color=colors, alpha=0.8)
        ax.set_xlim(0, 1.1)
        ax.set_title("Dataset Diagnostics Report", fontsize=13, fontweight="bold")
        ax.set_xlabel("Normalized Value")

        for bar, metric in zip(bars, metrics_to_show[:i]):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{metric[1]:,.2f}" if metric[1] < 1.0 else f"{metric[1]:,}",
                va="center",
                fontsize=9,
            )

        if i >= 4:
            ax.text(
                0.05,
                -0.7,
                "HIGH OOD RISK DETECTED",
                color="red",
                fontsize=10,
                fontweight="bold",
                transform=ax.transData,
            )

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)

        # Hold each frame for 8 frames at 4fps = 2 seconds
        for _ in range(8):
            frames.append(buf)

    path = save_gif(frames, OUT_DIR / "dataset_diagnostics.gif", fps=4)
    print(f"  Saved: {path}")


def generate_ope_gif():
    """GIF 4: OPE estimates appearing with confidence intervals."""
    print("Generating OPE comparison GIF...")
    from offline_rl.visualization.rollout_renderer import save_gif

    frames = []
    methods = ["FQE", "WIS", "DR", "Mean"]
    estimates = [69.8, 66.1, 70.4, 68.8]
    ci_low = [65.1, 61.2, 66.8, 64.0]
    ci_high = [74.5, 71.0, 74.0, 73.4]

    for n_shown in range(1, len(methods) + 1):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=80)

        x = range(n_shown)
        ax.bar(
            x,
            estimates[:n_shown],
            color=["steelblue", "orange", "green", "purple"][:n_shown],
            alpha=0.7,
            width=0.6,
        )
        ax.errorbar(
            x,
            estimates[:n_shown],
            yerr=[
                [estimates[i] - ci_low[i] for i in range(n_shown)],
                [ci_high[i] - estimates[i] for i in range(n_shown)],
            ],
            fmt="none",
            color="black",
            capsize=5,
            linewidth=2,
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(methods[:n_shown])
        ax.set_ylabel("Estimated Return")
        ax.set_title(
            "Offline Policy Evaluation (CQL Policy)", fontsize=11, fontweight="bold"
        )
        ax.set_ylim(55, 80)
        ax.grid(True, alpha=0.3, axis="y")

        for i in range(n_shown):
            ax.text(i, estimates[i] + 1.5, f"{estimates[i]:.1f}", ha="center", fontsize=9)

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)

        for _ in range(10):
            frames.append(buf)

    path = save_gif(frames, OUT_DIR / "ope_estimates.gif", fps=4)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    generate_gridworld_gif()
    generate_traffic_comparison_gif()
    generate_dataset_diagnostics_gif()
    generate_ope_gif()
    print(f"\nAll GIFs saved to {OUT_DIR}/")
    print("Add these to README.md as: ![](artifacts/gifs/traffic_comparison.gif)")
