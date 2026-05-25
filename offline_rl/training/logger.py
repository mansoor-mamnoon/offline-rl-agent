"""Training logger with JSON metrics and Rich console output."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class Logger:
    """Logs metrics to JSONL file and prints Rich tables."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._step = 0

        if _RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

        # Save git commit
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            commit = "unknown"
        (self.run_dir / "git_commit.txt").write_text(commit)

    def log(self, metrics: dict, step: int):
        """Append metrics to JSONL file."""
        entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._step = step

    def save_config(self, config: dict):
        """Save config as YAML."""
        try:
            import yaml
            config_path = self.run_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            import json
            config_path = self.run_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    def print_summary(self, metrics: dict):
        """Print a Rich table of metrics."""
        if _RICH_AVAILABLE and self.console is not None:
            table = Table(title=f"Step {self._step}", show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Value")
            for k, v in metrics.items():
                if isinstance(v, float):
                    table.add_row(k, f"{v:.4f}")
                else:
                    table.add_row(k, str(v))
            self.console.print(table)
        else:
            print(f"Step {self._step}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
