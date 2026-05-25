#!/usr/bin/env python
"""Standalone evaluation script for offline RL policies."""

import argparse
import json
import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_policy_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load policy callable from any checkpoint format."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", 32)
    act_dim = ckpt.get("act_dim", 4)

    from offline_rl.models.mlp import GaussianMLP, MLP

    if "policy" in ckpt:
        net = GaussianMLP(obs_dim, act_dim, [256, 256])
        net.load_state_dict(ckpt["policy"])
        net.eval()

        def select_action(obs):
            with torch.no_grad():
                mean, _ = net(torch.FloatTensor(obs).unsqueeze(0))
                return torch.tanh(mean).squeeze(0).numpy()

    elif "actor" in ckpt:
        net = MLP(obs_dim, act_dim, [256, 256], output_activation="tanh")
        net.load_state_dict(ckpt["actor"])
        net.eval()

        def select_action(obs):
            with torch.no_grad():
                return net(torch.FloatTensor(obs).unsqueeze(0)).squeeze(0).numpy()

    elif "model_state" in ckpt:
        net = GaussianMLP(obs_dim, act_dim, [256, 256])
        net.load_state_dict(ckpt["model_state"])
        net.eval()

        def select_action(obs):
            with torch.no_grad():
                mean, _ = net(torch.FloatTensor(obs).unsqueeze(0))
                return torch.tanh(mean).squeeze(0).numpy()

    else:
        raise ValueError(f"Cannot determine policy type from checkpoint keys: {list(ckpt.keys())}")

    return select_action, obs_dim, act_dim


def evaluate_in_env(policy, env, n_episodes=20, seed=0):
    """Run policy in environment and collect metrics."""
    np.random.seed(seed)
    returns = []
    slo_violations = []

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        ep_slo = 0
        ep_steps = 0
        done = False

        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            ep_slo += int(info.get("slo_violated", False))
            ep_steps += 1

        returns.append(total_reward)
        slo_violations.append(ep_slo / max(ep_steps, 1))

    returns = np.array(returns)
    n_worst = max(1, int(len(returns) * 0.05))

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "cvar_5pct": float(np.sort(returns)[:n_worst].mean()),
        "slo_violation_rate": float(np.mean(slo_violations)),
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained offline RL policy")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--env", default="traffic", choices=["traffic", "gridworld", "hospital"])
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ope", default="", help="Comma-separated OPE methods: fqe,wis,dr")
    parser.add_argument("--dataset", default=None, help="Dataset for OPE")
    parser.add_argument("--out", default=None, help="Save results to JSON")
    args = parser.parse_args()

    from glob import glob

    checkpoint = args.checkpoint
    if "*" in checkpoint:
        matches = sorted(glob(checkpoint))
        if matches:
            checkpoint = matches[-1]
        else:
            print(f"No checkpoint found: {args.checkpoint}")
            sys.exit(1)

    policy, obs_dim, act_dim = load_policy_from_checkpoint(checkpoint)

    if args.env == "traffic":
        from offline_rl.envs.traffic_routing import TrafficRoutingEnv

        env = TrafficRoutingEnv()
    elif args.env == "gridworld":
        from offline_rl.envs.gridworld import GridWorld, GridWorldConfig

        env = GridWorld(GridWorldConfig())
    elif args.env == "hospital":
        from offline_rl.envs.hospital_treatment import HospitalTreatmentEnv

        env = HospitalTreatmentEnv()

    print(f"\nEvaluating {Path(checkpoint).name} on {args.env} ({args.n_episodes} episodes)...")
    metrics = evaluate_in_env(policy, env, args.n_episodes, args.seed)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Mean Return:    {metrics['mean_return']:>10.2f}")
    print(f"  Std Return:     {metrics['std_return']:>10.2f}")
    print(f"  CVaR (5%):      {metrics['cvar_5pct']:>10.2f}")
    print(f"  SLO Viol Rate:  {metrics['slo_violation_rate']:>10.1%}")
    print("=" * 50)

    # OPE if dataset provided
    if args.ope and args.dataset:
        print("\nRunning offline policy evaluation...")
        from offline_rl.datasets.loader import load_dataset
        from offline_rl.evaluation.ope import OPEEvaluator

        dataset = load_dataset(args.dataset)
        evaluator = OPEEvaluator(dataset, behavior_policy=None)
        methods = [m.strip() for m in args.ope.split(",") if m.strip()]

        try:
            # Wrap policy in the interface OPEEvaluator expects
            class PolicyWrapper:
                def select_action(self, obs, deterministic=True):
                    return policy(obs)

            ope_result = evaluator.evaluate_policy(
                eval_policy=PolicyWrapper(),
                policy_name=Path(checkpoint).parent.name,
                methods=methods,
            )
            print(f"\n  {'Method':<10} {'Estimate':>10}  {'95% CI':>20}")
            print("  " + "-" * 45)
            if ope_result.fqe_estimate is not None:
                print(
                    f"  {'FQE':<10} {ope_result.fqe_estimate:>10.2f}  "
                    f"{str(ope_result.bootstrap_ci_95):>20}"
                )
            if ope_result.wis_estimate is not None:
                print(f"  {'WIS':<10} {ope_result.wis_estimate:>10.2f}")
            if ope_result.dr_estimate is not None:
                print(f"  {'DR':<10} {ope_result.dr_estimate:>10.2f}")
            metrics["ope"] = {
                "fqe": ope_result.fqe_estimate,
                "wis": ope_result.wis_estimate,
                "dr": ope_result.dr_estimate,
            }
        except Exception as e:
            print(f"  OPE failed: {e}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.out}")

    return metrics


if __name__ == "__main__":
    main()
