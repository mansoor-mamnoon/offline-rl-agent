#!/usr/bin/env python
"""Generate self-contained HTML report for a training run."""

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np


def encode_plotly_fig(fig) -> str:
    """Encode a plotly figure as base64 HTML string."""
    import plotly.io as pio

    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    return html


def load_run_metrics(run_dir: Path) -> list:
    metrics_file = run_dir / "metrics.jsonl"
    records = []
    if metrics_file.exists():
        with open(metrics_file) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def generate_report(run_dir: Path, out_path: Path):
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("plotly not installed; generating plain-HTML report.")
        go = None
        px = None

    # Load run data
    metrics = load_run_metrics(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.json"
    config_str = config_path.read_text() if config_path.exists() else "N/A"
    git_hash = (
        (run_dir / "git_commit.txt").read_text().strip()
        if (run_dir / "git_commit.txt").exists()
        else get_git_hash()
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate training curves
    steps = [r.get("step", i) for i, r in enumerate(metrics)]
    loss_values = [r.get("loss", r.get("q_loss", None)) for r in metrics]

    training_plot = ""
    if go is not None and any(v is not None for v in loss_values):
        valid = [(s, l) for s, l in zip(steps, loss_values) if l is not None]
        if valid:
            xs, ys = zip(*valid)
            fig = go.Figure(go.Scatter(x=list(xs), y=list(ys), name="Loss"))
            fig.update_layout(title="Training Loss", xaxis_title="Step", yaxis_title="Loss")
            training_plot = encode_plotly_fig(fig)

    # Summary stats
    final_metrics = metrics[-1] if metrics else {}

    def fmt_val(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    # HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OfflineRL-Lab Report: {run_dir.name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white;
             padding: 40px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 8px 0; font-size: 2em; }}
  .header .meta {{ opacity: 0.7; font-size: 0.9em; }}
  .card {{ background: white; border-radius: 12px; padding: 24px;
           margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .card h2 {{ margin-top: 0; color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 12px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
  .metric-box {{ background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }}
  .metric-box .value {{ font-size: 2em; font-weight: bold; color: #1a1a2e; }}
  .metric-box .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
  pre {{ background: #1a1a2e; color: #e0e0e0; padding: 16px; border-radius: 8px;
         overflow-x: auto; font-size: 0.85em; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #f8f9fa; font-weight: 600; }}
  .footer {{ text-align: center; padding: 24px; color: #999; font-size: 0.85em; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-size: 0.8em; font-weight: 600; }}
  .badge-green {{ background: #d4edda; color: #155724; }}
  .badge-red {{ background: #f8d7da; color: #721c24; }}
  .badge-yellow {{ background: #fff3cd; color: #856404; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>OfflineRL-Lab Report</h1>
    <div class="meta">
      Run: {run_dir.name} &nbsp;|&nbsp;
      Generated: {timestamp} &nbsp;|&nbsp;
      Commit: <code>{git_hash}</code>
    </div>
  </div>

  <div class="card">
    <h2>Training Summary</h2>
    <div class="metrics-grid">
      <div class="metric-box">
        <div class="value">{len(metrics):,}</div>
        <div class="label">Training Steps</div>
      </div>
      <div class="metric-box">
        <div class="value">{final_metrics.get('loss', final_metrics.get('q_loss', 'N/A'))}</div>
        <div class="label">Final Loss</div>
      </div>
      <div class="metric-box">
        <div class="value">{final_metrics.get('mean_return', 'N/A')}</div>
        <div class="label">Mean Return</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Training Curves</h2>
    {training_plot if training_plot else '<p>No training data available.</p>'}
  </div>

  <div class="card">
    <h2>All Metrics (Final Step)</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      {''.join(f"<tr><td>{k}</td><td>{fmt_val(v)}</td></tr>" for k, v in final_metrics.items())}
    </table>
  </div>

  <div class="card">
    <h2>Configuration</h2>
    <pre>{config_str}</pre>
  </div>

  <div class="footer">
    Generated by OfflineRL-Lab &nbsp;|&nbsp; {timestamp} &nbsp;|&nbsp; commit {git_hash}
  </div>
</div>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Report saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report for a training run")
    parser.add_argument("--run", required=True, help="Path to run directory")
    parser.add_argument("--out", default="report.html", help="Output HTML path")
    args = parser.parse_args()

    run_dir = Path(args.run)
    # Handle glob pattern
    if "*" in str(run_dir):
        from glob import glob

        matches = glob(str(run_dir))
        if matches:
            run_dir = Path(sorted(matches)[-1])
        else:
            print(f"No runs found matching {args.run}")
            sys.exit(1)

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    generate_report(run_dir, Path(args.out))


if __name__ == "__main__":
    main()
