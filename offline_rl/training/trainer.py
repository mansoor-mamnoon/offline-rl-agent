"""Training loop for offline RL algorithms."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional

from offline_rl.datasets.replay_buffer import ReplayBuffer
from offline_rl.training.logger import Logger


class BCTrainer:
    """Training loop for Behavior Cloning."""

    def __init__(
        self,
        algorithm,
        dataset: dict,
        config,
        logger: Logger,
    ):
        self.algorithm = algorithm
        self.config = config
        self.logger = logger

        # Build replay buffer
        obs_dim = dataset["observations"].shape[1]
        actions = dataset["actions"]
        act_dim = actions.shape[1] if actions.ndim > 1 else 1

        n = len(dataset["observations"])
        self.buffer = ReplayBuffer(
            capacity=n,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=algorithm.device,
        )
        self.buffer.load_from_dataset(dataset)

    def train(self) -> dict:
        """Run n_epochs training epochs.

        Returns dict of final metrics.
        """
        n_epochs = self.config.n_epochs
        batch_size = self.config.batch_size

        # Estimate steps per epoch (at least 1)
        steps_per_epoch = max(1, len(self.buffer) // batch_size)

        all_losses = []
        step = 0

        for epoch in range(n_epochs):
            epoch_losses = []
            for _ in range(steps_per_epoch):
                batch = self.buffer.sample(batch_size)
                metrics = self.algorithm.train_step(batch)
                epoch_losses.append(metrics["loss"])
                step += 1

            epoch_loss = float(np.mean(epoch_losses))
            all_losses.append(epoch_loss)

            log_metrics = {"epoch": epoch, "loss": epoch_loss}
            self.logger.log(log_metrics, step=epoch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.logger.print_summary(log_metrics)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                ckpt_path = str(self.logger.run_dir / f"checkpoint_epoch{epoch+1}.pt")
                self.algorithm.save(ckpt_path)

        # Save final checkpoint
        final_path = str(self.logger.run_dir / "checkpoint.pt")
        self.algorithm.save(final_path)

        return {"final_loss": all_losses[-1], "losses": all_losses}

    def evaluate(self, env, n_episodes: int = 10) -> dict:
        """Run policy in environment for n_episodes.

        Returns dict with mean_return, std_return.
        """
        returns = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                action = self.algorithm.select_action(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_return += reward
            returns.append(ep_return)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
        }
