#!/usr/bin/env python
"""End-to-end training script for offline RL algorithms."""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train an offline RL algorithm")
    parser.add_argument("--algo", default="bc", choices=["bc", "cql", "iql", "td3bc", "dt"])
    parser.add_argument("--env", default="traffic", choices=["traffic", "gridworld"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-dataset-episodes", type=int, default=200)
    parser.add_argument("--n-train-epochs", type=int, default=5)
    parser.add_argument("--out", default="runs/")
    parser.add_argument("--dataset", default=None, help="Path to existing HDF5 dataset")
    args = parser.parse_args()

    # 1. Set seed
    set_seed(args.seed)
    print(f"[train] algo={args.algo}, env={args.env}, seed={args.seed}")

    # 2. Create/load environment
    if args.env == "traffic":
        from offline_rl.envs.traffic_routing import TrafficRoutingEnv, load_aware_policy
        env = TrafficRoutingEnv(seed=args.seed)
        policy_fn = load_aware_policy
        obs_dim = 32
        act_dim = 4
        action_space = "continuous"
    else:
        from offline_rl.envs.gridworld import GridWorld, GridWorldConfig, mixed_policy
        env_cfg = GridWorldConfig()
        env = GridWorld(env_cfg)
        policy_fn = mixed_policy
        obs_dim = 2
        act_dim = 4
        action_space = "discrete"

    # 3. Generate dataset if not provided
    if args.dataset is None:
        print(f"[train] Generating {args.n_dataset_episodes} episodes...")
        dataset = env.generate_dataset(policy_fn, n_episodes=args.n_dataset_episodes)
    else:
        from offline_rl.datasets.loader import load_dataset
        print(f"[train] Loading dataset from {args.dataset}")
        dataset = load_dataset(args.dataset)

    n_transitions = len(dataset["observations"])
    print(f"[train] Dataset: {n_transitions} transitions")

    # 4. Create algorithm
    if args.algo == "bc":
        from offline_rl.algorithms.bc import BehaviorCloning, BCConfig
        from offline_rl.training.trainer import BCTrainer
        algo_cfg = BCConfig(
            hidden_dims=[256, 256],
            lr=3e-4,
            batch_size=256,
            n_epochs=args.n_train_epochs,
            action_space=action_space,
        )
        algorithm = BehaviorCloning(obs_dim, act_dim, algo_cfg)
        TrainerClass = BCTrainer

    elif args.algo == "cql":
        from offline_rl.algorithms.cql import CQL, CQLConfig
        from offline_rl.training.trainer import CQLTrainer
        algo_cfg = CQLConfig(
            hidden_dims=[256, 256],
            lr=3e-4,
            batch_size=256,
            n_epochs=args.n_train_epochs,
            alpha=5.0,
            alpha_auto=True,
        )
        algorithm = CQL(obs_dim, act_dim, algo_cfg)
        TrainerClass = CQLTrainer

    elif args.algo == "iql":
        from offline_rl.algorithms.iql import IQL, IQLConfig
        from offline_rl.training.trainer import IQLTrainer
        algo_cfg = IQLConfig(
            hidden_dims=[256, 256],
            lr=3e-4,
            batch_size=256,
            n_epochs=args.n_train_epochs,
        )
        algorithm = IQL(obs_dim, act_dim, algo_cfg)
        TrainerClass = IQLTrainer

    elif args.algo == "td3bc":
        from offline_rl.algorithms.td3_bc import TD3BC, TD3BCConfig
        from offline_rl.training.trainer import TD3BCTrainer
        algo_cfg = TD3BCConfig(
            hidden_dims=[256, 256],
            lr=3e-4,
            batch_size=256,
            n_epochs=args.n_train_epochs,
        )
        algorithm = TD3BC(obs_dim, act_dim, algo_cfg)
        TrainerClass = TD3BCTrainer

    else:
        print(f"[train] WARNING: {args.algo} not yet implemented. Using BC.")
        from offline_rl.algorithms.bc import BehaviorCloning, BCConfig
        from offline_rl.training.trainer import BCTrainer
        algo_cfg = BCConfig(
            hidden_dims=[256, 256],
            lr=3e-4,
            batch_size=256,
            n_epochs=args.n_train_epochs,
            action_space=action_space,
        )
        algorithm = BehaviorCloning(obs_dim, act_dim, algo_cfg)
        TrainerClass = BCTrainer

    # 5. Create logger
    from offline_rl.training.logger import Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = str(Path(args.out) / f"{args.algo}_{args.env}_{timestamp}")
    logger = Logger(run_dir)

    config_dict = {
        "algo": args.algo,
        "env": args.env,
        "seed": args.seed,
        "n_dataset_episodes": args.n_dataset_episodes,
        "n_train_epochs": args.n_train_epochs,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }
    logger.save_config(config_dict)

    # 6. Create trainer and run
    trainer = TrainerClass(algorithm, dataset, algo_cfg, logger)

    print(f"[train] Starting training for {args.n_train_epochs} epochs...")
    results = trainer.train()

    # 7. Confirm output files
    final_ckpt = Path(run_dir) / "checkpoint.pt"
    print(f"[train] Training complete!")
    print(f"[train] Final loss: {results['final_loss']:.4f}")
    print(f"[train] Checkpoint saved: {final_ckpt}")
    print(f"[train] Run dir: {run_dir}")

    assert final_ckpt.exists(), f"Checkpoint not found: {final_ckpt}"
    assert (Path(run_dir) / "metrics.jsonl").exists(), "metrics.jsonl not found"
    assert (Path(run_dir) / "config.yaml").exists() or (
        Path(run_dir) / "config.json"
    ).exists(), "config not found"

    print("[train] All output files verified.")


if __name__ == "__main__":
    main()
