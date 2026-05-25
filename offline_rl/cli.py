import click
import os
import sys
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path


@click.group()
def cli():
    """OfflineRL-Lab: offline reinforcement learning with safety constraints."""
    pass


@cli.command()
@click.option("--algo", default="bc", type=click.Choice(["bc", "cql", "iql", "td3bc", "dt"]))
@click.option("--env", default="traffic", type=click.Choice(["traffic", "gridworld"]))
@click.option("--dataset", default=None, help="Path to HDF5 dataset (generated if not given)")
@click.option("--config", default=None, help="Path to algorithm config YAML")
@click.option("--seed", default=42, type=int)
@click.option("--out", default="runs/")
@click.option("--n-dataset-episodes", default=500, type=int)
@click.option("--n-train-epochs", default=100, type=int)
def train(algo, env, dataset, config, seed, out, n_dataset_episodes, n_train_epochs):
    """Train an offline RL algorithm."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    click.echo(f"[orl] Training {algo} on {env} (seed={seed})")

    # Create environment
    if env == "traffic":
        from offline_rl.envs.traffic_routing import TrafficRoutingEnv, load_aware_policy
        environment = TrafficRoutingEnv(seed=seed)
        policy_fn = load_aware_policy
        obs_dim = 32
        act_dim = 4
        action_space = "continuous"
    else:
        from offline_rl.envs.gridworld import GridWorld, GridWorldConfig, mixed_policy
        cfg = GridWorldConfig()
        environment = GridWorld(cfg)
        policy_fn = mixed_policy
        obs_dim = 2
        act_dim = 4
        action_space = "discrete"

    # Generate or load dataset
    if dataset is None:
        click.echo(f"[orl] Generating {n_dataset_episodes} episodes...")
        raw_dataset = environment.generate_dataset(policy_fn, n_episodes=n_dataset_episodes)
    else:
        from offline_rl.datasets.loader import load_dataset
        click.echo(f"[orl] Loading dataset from {dataset}")
        raw_dataset = load_dataset(dataset)

    click.echo(f"[orl] Dataset: {len(raw_dataset['observations'])} transitions")

    # Build algorithm config
    if algo == "bc":
        from offline_rl.algorithms.bc import BehaviorCloning, BCConfig
        algo_cfg = BCConfig(n_epochs=n_train_epochs, action_space=action_space)
        algorithm = BehaviorCloning(obs_dim, act_dim, algo_cfg)
    else:
        click.echo(f"[orl] Algorithm '{algo}' not yet implemented, using BC fallback.")
        from offline_rl.algorithms.bc import BehaviorCloning, BCConfig
        algo_cfg = BCConfig(n_epochs=n_train_epochs, action_space=action_space)
        algorithm = BehaviorCloning(obs_dim, act_dim, algo_cfg)

    # Set up logger
    from offline_rl.training.logger import Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = str(Path(out) / f"{algo}_{env}_{timestamp}")
    logger = Logger(run_dir)

    config_dict = {
        "algo": algo,
        "env": env,
        "seed": seed,
        "n_dataset_episodes": n_dataset_episodes,
        "n_train_epochs": n_train_epochs,
    }
    logger.save_config(config_dict)

    # Train
    from offline_rl.training.trainer import BCTrainer
    trainer = BCTrainer(algorithm, raw_dataset, algo_cfg, logger)
    results = trainer.train()

    click.echo(f"[orl] Training complete. Final loss: {results['final_loss']:.4f}")
    click.echo(f"[orl] Run saved to: {run_dir}")
