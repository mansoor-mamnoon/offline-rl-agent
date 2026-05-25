#!/usr/bin/env python
"""Run all algorithms and print benchmark comparison table."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_algorithm(
    algo: str,
    env: str,
    seed: int,
    n_epochs: int,
    n_dataset_episodes: int = 200,
) -> dict:
    """Run a single algorithm and parse metrics from its output."""
    python = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")
    train_script = str(Path(__file__).parent / "train.py")
    cmd = [
        python,
        train_script,
        "--algo",
        algo,
        "--env",
        env,
        "--seed",
        str(seed),
        "--n-train-epochs",
        str(n_epochs),
        "--n-dataset-episodes",
        str(n_dataset_episodes),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=300,
        )
        if result.returncode != 0:
            print(f"  [WARN] {algo}/seed={seed} exited with code {result.returncode}")
            return {
                "mean_return": float("nan"),
                "final_loss": float("nan"),
                "error": result.stderr[-200:],
            }

        # Parse final loss from stdout
        final_loss = float("nan")
        for line in result.stdout.splitlines():
            if "Final loss:" in line:
                try:
                    final_loss = float(line.split("Final loss:")[-1].strip())
                except ValueError:
                    pass

        # Find the run dir from stdout and read metrics.jsonl
        run_dir = None
        for line in result.stdout.splitlines():
            if "Run dir:" in line:
                run_dir = line.split("Run dir:")[-1].strip()
                break

        mean_return = float("nan")
        if run_dir and Path(run_dir).exists():
            metrics_path = Path(run_dir) / "metrics.jsonl"
            if metrics_path.exists():
                records = []
                with open(metrics_path) as f:
                    for line in f:
                        try:
                            records.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            pass
                if records:
                    last = records[-1]
                    mean_return = last.get("mean_return", float("nan"))
                    final_loss = last.get("q_loss", last.get("loss", final_loss))

        return {
            "mean_return": mean_return,
            "final_loss": final_loss,
            "run_dir": run_dir or "",
        }

    except subprocess.TimeoutExpired:
        print(f"  [WARN] {algo}/seed={seed} timed out")
        return {"mean_return": float("nan"), "final_loss": float("nan"), "error": "timeout"}
    except Exception as e:
        print(f"  [WARN] {algo}/seed={seed} failed: {e}")
        return {"mean_return": float("nan"), "final_loss": float("nan"), "error": str(e)}


def format_val(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  N/A  "
    return f"{val:7.3f}"


def main():
    parser = argparse.ArgumentParser(description="Reproduce benchmark table for OfflineRL-Lab")
    parser.add_argument("--env", default="traffic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--n-dataset-episodes", type=int, default=200)
    parser.add_argument("--algos", nargs="+", default=["bc", "cql", "iql", "td3bc"])
    args = parser.parse_args()

    print(f"\nReproducing benchmark: env={args.env}, seeds={args.seeds}, epochs={args.n_epochs}")
    print("=" * 60)

    results = {}
    for algo in args.algos:
        print(f"\nRunning {algo.upper()}...")
        seed_results = []
        for seed in args.seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            metrics = run_algorithm(algo, args.env, seed, args.n_epochs, args.n_dataset_episodes)
            seed_results.append(metrics)
            ret = metrics.get("mean_return", float("nan"))
            loss = metrics.get("final_loss", float("nan"))
            print(f"loss={loss:.4f}" if np.isfinite(loss) else "FAILED")
        results[algo] = seed_results

    # Print table
    print("\n")
    print("Algorithm    Loss (mean±std)     Return (mean±std)")
    print("─" * 55)
    for algo, seed_results in results.items():
        losses = [
            r.get("final_loss", float("nan"))
            for r in seed_results
            if np.isfinite(r.get("final_loss", float("nan")))
        ]
        returns = [
            r.get("mean_return", float("nan"))
            for r in seed_results
            if np.isfinite(r.get("mean_return", float("nan")))
        ]

        loss_str = f"{np.mean(losses):.4f}±{np.std(losses):.4f}" if losses else "   N/A   "
        return_str = f"{np.mean(returns):.3f}±{np.std(returns):.3f}" if returns else "   N/A   "
        print(f"{algo:<12} {loss_str:<20} {return_str}")

    # Save to artifacts/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(__file__).parent.parent
        / "artifacts"
        / "benchmark_tables"
        / f"traffic_safety_{timestamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "env": args.env,
        "seeds": args.seeds,
        "n_epochs": args.n_epochs,
        "timestamp": timestamp,
        "results": {
            algo: [{k: v for k, v in r.items() if k != "run_dir"} for r in seed_results]
            for algo, seed_results in results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
