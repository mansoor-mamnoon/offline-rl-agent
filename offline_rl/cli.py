import click
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
@click.option("--dataset", required=True, help="Path to HDF5 dataset file")
@click.option("--out", default=None, help="Optional path to save JSON report")
def diagnose(dataset, out):
    """Run dataset diagnostics and print a formatted report."""
    try:
        from rich.console import Console
        from rich.table import Table

        _rich = True
    except ImportError:
        _rich = False

    from offline_rl.datasets.loader import load_dataset
    from offline_rl.datasets.diagnostics import DatasetDiagnostics

    dataset_path = Path(dataset)
    click.echo(f"[orl] Loading dataset from {dataset_path}")
    raw_dataset = load_dataset(str(dataset_path))

    click.echo("[orl] Running diagnostics (this may take a moment)...")
    diag = DatasetDiagnostics(raw_dataset)
    report = diag.run_all()

    if _rich:
        console = Console()
        table = Table(
            title=f"Dataset: {dataset_path.name}",
            show_header=True,
            header_style="bold cyan",
            expand=False,
        )
        table.add_column("Metric", style="dim", width=28)
        table.add_column("Value")

        table.add_row("Transitions", f"{report.n_transitions:,}")
        table.add_row("Episodes", f"{report.n_episodes:,}")
        table.add_row("Obs dim", str(report.obs_dim))
        table.add_row("Act dim", str(report.act_dim))
        table.add_row("", "")
        table.add_row("Coverage score", f"{report.coverage_score:.3f}")
        table.add_row("Behavior entropy", f"{report.behavior_entropy:.3f}")
        table.add_row("OOD risk", report.ood_risk["risk_level"].upper())
        table.add_row("Reward skew", f"{report.reward_stats['skewness']:.2f}")
        table.add_row(
            "Terminal rate",
            f"{report.episode_stats['terminal_rate']*100:.1f}%",
        )
        table.add_row("Support mismatch risk", f"{report.support_mismatch_risk:.3f}")

        console.print(table)

        # Warnings
        warnings = []
        if report.ood_risk["risk_level"] == "high":
            frac = report.ood_risk["sparse_state_fraction"]
            warnings.append(f"{frac*100:.1f}% of random actions are far from dataset support")
        if abs(report.reward_stats["skewness"]) > 2.0:
            warnings.append(
                f"Reward distribution heavily skewed "
                f"(skewness={report.reward_stats['skewness']:.2f})"
            )
        if report.episode_stats["terminal_rate"] < 0.1:
            warnings.append("Very few episodes reach terminal state — check done flags")

        if warnings:
            console.print("[bold yellow]Warnings:[/bold yellow]")
            for w in warnings:
                console.print(f"  [yellow]· {w}[/yellow]")
    else:
        click.echo(report.summary_string())

    if out:
        import json

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        click.echo(f"[orl] Report saved to {out_path}")


@cli.command()
@click.option("--algo", default="bc", type=click.Choice(["bc", "cql", "iql", "td3bc", "dt", "awac"]))
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
        from offline_rl.training.trainer import BCTrainer

        algo_cfg = BCConfig(n_epochs=n_train_epochs, action_space=action_space)
        algorithm = BehaviorCloning(obs_dim, act_dim, algo_cfg)
        TrainerClass = BCTrainer
    elif algo == "cql":
        from offline_rl.algorithms.cql import CQL, CQLConfig
        from offline_rl.training.trainer import CQLTrainer

        algo_cfg = CQLConfig(n_epochs=n_train_epochs)
        algorithm = CQL(obs_dim, act_dim, algo_cfg)
        TrainerClass = CQLTrainer
    elif algo == "iql":
        from offline_rl.algorithms.iql import IQL, IQLConfig
        from offline_rl.training.trainer import IQLTrainer

        algo_cfg = IQLConfig(n_epochs=n_train_epochs)
        algorithm = IQL(obs_dim, act_dim, algo_cfg)
        TrainerClass = IQLTrainer
    elif algo == "td3bc":
        from offline_rl.algorithms.td3_bc import TD3BC, TD3BCConfig
        from offline_rl.training.trainer import TD3BCTrainer

        algo_cfg = TD3BCConfig(n_epochs=n_train_epochs)
        algorithm = TD3BC(obs_dim, act_dim, algo_cfg)
        TrainerClass = TD3BCTrainer
    elif algo == "awac":
        from offline_rl.algorithms.awac import AWAC, AWACConfig
        from offline_rl.training.trainer import AWACTrainer

        algo_cfg = AWACConfig(n_epochs=n_train_epochs)
        algorithm = AWAC(obs_dim, act_dim, algo_cfg)
        TrainerClass = AWACTrainer
    else:
        click.echo(f"[orl] Algorithm '{algo}' not yet implemented, using BC fallback.")
        from offline_rl.algorithms.bc import BehaviorCloning, BCConfig
        from offline_rl.training.trainer import BCTrainer

        algo_cfg = BCConfig(n_epochs=n_train_epochs, action_space=action_space)
        algorithm = BehaviorCloning(obs_dim, act_dim, algo_cfg)
        TrainerClass = BCTrainer

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
    trainer = TrainerClass(algorithm, raw_dataset, algo_cfg, logger)
    results = trainer.train()

    click.echo(f"[orl] Training complete. Final loss: {results['final_loss']:.4f}")
    click.echo(f"[orl] Run saved to: {run_dir}")


@cli.command()
@click.option("--checkpoint", required=True, help="Path to checkpoint .pt file")
@click.option("--dataset", required=True, help="Path to HDF5 dataset file")
@click.option("--ope", default="fqe,wis,dr", help="Comma-separated OPE methods")
@click.option("--algo", default="cql", type=click.Choice(["bc", "cql", "iql", "td3bc"]))
def evaluate(checkpoint, dataset, ope, algo):
    """Evaluate a policy checkpoint using offline policy evaluation methods."""
    methods = [m.strip() for m in ope.split(",")]
    click.echo(f"[orl] Evaluating checkpoint: {checkpoint}")
    click.echo(f"[orl] Dataset: {dataset}")
    click.echo(f"[orl] Methods: {methods}")

    from offline_rl.datasets.loader import load_dataset
    from offline_rl.evaluation.ope import OPEEvaluator

    raw_dataset = load_dataset(dataset)
    click.echo(f"[orl] Dataset: {len(raw_dataset['observations'])} transitions")

    # Load policy
    if algo == "bc":
        from offline_rl.algorithms.bc import BehaviorCloning

        policy = BehaviorCloning.load(checkpoint)
    elif algo == "cql":
        from offline_rl.algorithms.cql import CQL

        policy = CQL.load(checkpoint)
    elif algo == "iql":
        from offline_rl.algorithms.iql import IQL

        policy = IQL.load(checkpoint)
    elif algo == "td3bc":
        from offline_rl.algorithms.td3_bc import TD3BC

        policy = TD3BC.load(checkpoint)
    else:
        raise click.BadParameter(f"Unknown algo: {algo}")

    evaluator = OPEEvaluator(raw_dataset)
    result = evaluator.evaluate_policy(policy, policy_name=algo, methods=methods)

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"OPE Results: {algo}", header_style="bold cyan")
        table.add_column("Method", style="dim")
        table.add_column("Estimate")
        if result.fqe_estimate is not None:
            table.add_row("FQE", f"{result.fqe_estimate:.4f}")
        if result.wis_estimate is not None:
            table.add_row("WIS", f"{result.wis_estimate:.4f}")
        if result.dr_estimate is not None:
            table.add_row("DR", f"{result.dr_estimate:.4f}")
        lo, hi = result.bootstrap_ci_95
        table.add_row("Bootstrap CI 95%", f"[{lo:.4f}, {hi:.4f}]")
        console.print(table)
    except ImportError:
        click.echo(f"FQE:  {result.fqe_estimate}")
        click.echo(f"WIS:  {result.wis_estimate}")
        click.echo(f"DR:   {result.dr_estimate}")
        click.echo(f"CI95: {result.bootstrap_ci_95}")


@cli.command()
@click.option("--run", required=True, help="Path to run directory")
@click.option("--out", default="report.html")
def report(run, out):
    """Generate self-contained HTML report for a training run."""
    import subprocess

    report_script = Path(__file__).parent.parent / "scripts" / "generate_report.py"
    subprocess.run([sys.executable, str(report_script), "--run", run, "--out", out])


@cli.command()
@click.option("--port", default=8501, type=int)
def dashboard(port):
    """Launch the OfflineRL-Lab Streamlit dashboard."""
    import subprocess

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", str(port)]
    )


@cli.command()
@click.option("--checkpoint", required=True, help="Path to checkpoint.pt")
@click.option(
    "--env",
    default="traffic",
    type=click.Choice(["traffic", "gridworld", "cliffwalking"]),
)
@click.option("--out", default="artifacts/gifs/rollout.gif", help="Output GIF path")
@click.option("--n-steps", default=100, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--fps", default=10, type=int)
@click.option(
    "--compare",
    is_flag=True,
    default=False,
    help="Compare multiple policies",
)
def render(checkpoint, env, out, n_steps, seed, fps, compare):
    """Generate a rollout GIF from a trained policy checkpoint."""
    import os
    import torch

    from offline_rl.visualization.rollout_renderer import collect_rollout, save_gif

    # Load environment
    if env == "traffic":
        from offline_rl.envs.traffic_routing import TrafficRoutingEnv

        environment = TrafficRoutingEnv()
    elif env == "gridworld":
        from offline_rl.envs.gridworld import GridWorld, GridWorldConfig

        environment = GridWorld(GridWorldConfig())
    elif env == "cliffwalking":
        from offline_rl.envs.cliffwalking import make_cliff_walking

        environment = make_cliff_walking()

    # Resolve glob in checkpoint path
    ckpt_path = Path(checkpoint)
    if "*" in str(ckpt_path):
        from glob import glob

        matches = sorted(glob(str(ckpt_path)))
        if matches:
            ckpt_path = Path(matches[-1])
        else:
            click.echo(f"No checkpoint found matching: {checkpoint}")
            return

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Reconstruct policy
    obs_dim = ckpt.get("obs_dim", 32)
    act_dim = ckpt.get("act_dim", 4)

    from offline_rl.models.mlp import GaussianMLP

    policy_net = GaussianMLP(obs_dim, act_dim, [256, 256])
    if "policy" in ckpt:
        policy_net.load_state_dict(ckpt["policy"])
    elif "model_state" in ckpt:
        policy_net.load_state_dict(ckpt["model_state"])
    elif "actor" in ckpt:
        # For algorithms that store actor instead of policy
        try:
            policy_net.load_state_dict(ckpt["actor"])
        except Exception:
            click.echo("Warning: could not load actor weights, using random policy")
    policy_net.eval()

    def policy_fn(obs):
        with torch.no_grad():
            mean, _ = policy_net(torch.FloatTensor(obs).unsqueeze(0))
            return torch.tanh(mean).squeeze(0).numpy()

    click.echo(f"Collecting rollout ({n_steps} steps)...")
    np.random.seed(seed)
    frames, metrics = collect_rollout(environment, policy_fn, max_steps=n_steps, seed=seed)

    out_path = save_gif(frames, out, fps=fps)
    click.echo(f"GIF saved: {out_path}")
    click.echo(f"Return: {metrics['total_return']:.2f} over {metrics['n_steps']} steps")


@cli.command()
@click.option("--runs", multiple=True, required=True, help="Run directories to compare")
@click.option("--metric", default="loss", help="Metric to compare")
@click.option("--out", default=None, help="Save comparison plot to path")
def compare(runs, metric, out):
    """Compare multiple training runs side by side."""
    import json

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        _rich = True
    except ImportError:
        _rich = False

    all_metrics = {}
    for run_path in runs:
        p = Path(run_path)
        if not p.exists():
            click.echo(f"Warning: {run_path} not found, skipping")
            continue
        metrics_file = p / "metrics.jsonl"
        if not metrics_file.exists():
            click.echo(f"Warning: no metrics.jsonl in {run_path}")
            continue
        records = []
        with open(metrics_file) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        all_metrics[p.name] = records

    if not all_metrics:
        click.echo("No valid runs found.")
        return

    # Print comparison table
    if _rich:
        table = Table(title="Run Comparison", show_header=True)
        table.add_column("Run", style="bold cyan")
        table.add_column("Steps", justify="right")
        table.add_column("Final Loss", justify="right")
        table.add_column("Mean Return", justify="right")
        table.add_column("Best Loss", justify="right")

        for run_name, records in all_metrics.items():
            if not records:
                continue
            final = records[-1]
            losses = [
                r.get("loss", r.get("q_loss"))
                for r in records
                if r.get("loss") is not None or r.get("q_loss") is not None
            ]
            returns = [
                r.get("mean_return") for r in records if r.get("mean_return") is not None
            ]

            final_loss_val = final.get("loss", final.get("q_loss"))
            final_loss_str = (
                f"{final_loss_val:.4f}" if isinstance(final_loss_val, float) else "N/A"
            )

            table.add_row(
                run_name,
                str(len(records)),
                final_loss_str,
                f"{returns[-1]:.2f}" if returns else "N/A",
                f"{min(losses):.4f}" if losses else "N/A",
            )
        console.print(table)
    else:
        for run_name, records in all_metrics.items():
            if records:
                final = records[-1]
                click.echo(
                    f"{run_name}: steps={len(records)}, final_loss={final.get('loss', 'N/A')}"
                )

    # Optional plot
    if out:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            for run_name, records in all_metrics.items():
                steps = list(range(len(records)))
                values = [r.get("loss", r.get("q_loss")) for r in records]
                valid = [(s, v) for s, v in zip(steps, values) if v is not None]
                if valid:
                    xs, ys = zip(*valid)
                    ax.plot(xs, ys, label=run_name, linewidth=2)

            ax.set_xlabel("Step")
            ax.set_ylabel(metric)
            ax.set_title("Training Run Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            click.echo(f"Plot saved to {out}")
        except Exception as e:
            click.echo(f"Could not save plot: {e}")
