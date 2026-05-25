#!/usr/bin/env python
"""Standalone dataset diagnostics script."""
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Diagnose an offline RL dataset")
    parser.add_argument("--dataset", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--out", default=None, help="Save diagnostics JSON to path")
    parser.add_argument("--plot", default=None, help="Save plots to directory")
    args = parser.parse_args()

    from offline_rl.datasets.loader import load_dataset, dataset_info
    from offline_rl.datasets.diagnostics import DatasetDiagnostics

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    info = dataset_info(dataset)

    print(f"\nRunning diagnostics on {info['n_transitions']:,} transitions...")
    diag = DatasetDiagnostics(dataset)
    report = diag.run_all()

    risk_symbols = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]"}
    risk = report.ood_risk.get("risk_level", "unknown")

    print("\n" + "=" * 45)
    print(f"  Dataset: {Path(args.dataset).name}")
    print("=" * 45)
    print(f"  Transitions:  {report.n_transitions:>12,}")
    print(f"  Episodes:     {report.n_episodes:>12,}")
    print(f"  Obs dim:      {report.obs_dim:>12}")
    print(f"  Act dim:      {report.act_dim:>12}")
    print()
    print(f"  Coverage:     {report.coverage_score:>12.3f}")
    print(f"  Entropy:      {report.behavior_entropy:>12.3f}")
    print(f"  OOD Risk:     {risk_symbols.get(risk, '?')} {risk.upper():>10}")
    print(f"  Reward skew:  {report.reward_stats['skewness']:>12.3f}")
    print(f"  Terminal rate:{report.episode_stats['terminal_rate']:>12.1%}")
    print(f"  Outliers:     {len(report.outlier_indices):>12,} transitions")
    print("=" * 45)

    # Warnings
    warnings = []
    if report.ood_risk.get("risk_level") == "high":
        frac = report.ood_risk.get("sparse_state_fraction", 0)
        warnings.append(
            f"{frac*100:.1f}% of random actions are far from dataset support"
        )
    if abs(report.reward_stats.get("skewness", 0)) > 2:
        warnings.append(
            f"Reward distribution is heavily skewed "
            f"(skew={report.reward_stats['skewness']:.2f})"
        )
    if len(report.outlier_indices) > info["n_transitions"] * 0.01:
        warnings.append(
            f"{len(report.outlier_indices)} outlier transitions detected (>1%)"
        )

    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"  . {w}")
        print()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_dict = report.to_dict()
        with open(out_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"Diagnostics saved to {args.out}")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            plot_dir = Path(args.plot)
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Reward histogram
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(
                dataset["rewards"].flatten(),
                bins=50,
                color="steelblue",
                alpha=0.7,
                edgecolor="white",
            )
            ax.set_xlabel("Reward")
            ax.set_ylabel("Count")
            ax.set_title("Reward Distribution")
            ax.axvline(
                report.reward_stats["mean"],
                color="red",
                linestyle="--",
                label=f'Mean={report.reward_stats["mean"]:.2f}',
            )
            ax.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "reward_hist.png", dpi=150)
            plt.close()

            # PCA scatter
            if report.state_pca_points is not None:
                pts = np.array(report.state_pca_points)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    alpha=0.3,
                    s=2,
                    c=dataset["rewards"][: len(pts)].flatten(),
                    cmap="coolwarm",
                )
                ax.set_title(
                    f'State PCA (var={report.pca_explained_variance:.2%})'
                )
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.tight_layout()
                plt.savefig(plot_dir / "state_pca.png", dpi=150)
                plt.close()

            print(f"Plots saved to {args.plot}/")
        except Exception as e:
            print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()
