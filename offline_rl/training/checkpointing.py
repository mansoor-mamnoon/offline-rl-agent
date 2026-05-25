"""Checkpoint management: save, load, resume training."""

import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
import torch
import numpy as np


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def compute_dataset_hash(dataset: dict) -> str:
    """Compute a reproducible hash of the dataset."""
    h = hashlib.md5()
    for key in sorted(dataset.keys()):
        arr = dataset[key]
        if hasattr(arr, "tobytes"):
            # Hash first 1000 elements for speed
            h.update(arr.flatten()[:1000].astype(np.float32).tobytes())
        h.update(key.encode())
    return h.hexdigest()[:12]


class CheckpointManager:
    """Manage model checkpoints with versioning and resume support."""

    def __init__(self, run_dir: str, keep_n: int = 3):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n
        self._checkpoint_log = []

    def save(self, algorithm, step: int, metrics: dict, is_best: bool = False) -> str:
        """Save a checkpoint. Returns path."""
        ckpt_path = self.run_dir / f"checkpoint_step{step:06d}.pt"

        save_dict = {
            "step": step,
            "metrics": metrics,
            "git_hash": get_git_hash(),
            "timestamp": datetime.now().isoformat(),
        }

        # Collect algorithm state
        if hasattr(algorithm, "qf"):
            save_dict["qf"] = algorithm.qf.state_dict()
            save_dict["qf_target"] = (
                algorithm.qf_target.state_dict() if hasattr(algorithm, "qf_target") else None
            )
        if hasattr(algorithm, "policy"):
            save_dict["policy"] = algorithm.policy.state_dict()
        if hasattr(algorithm, "actor"):
            save_dict["actor"] = algorithm.actor.state_dict()
        if hasattr(algorithm, "model"):
            save_dict["model"] = algorithm.model.state_dict()
        if hasattr(algorithm, "config"):
            save_dict["config"] = algorithm.config
        if hasattr(algorithm, "obs_dim"):
            save_dict["obs_dim"] = algorithm.obs_dim
        if hasattr(algorithm, "act_dim"):
            save_dict["act_dim"] = algorithm.act_dim

        torch.save(save_dict, ckpt_path)

        # Also save as checkpoint.pt (latest)
        latest_path = self.run_dir / "checkpoint.pt"
        torch.save(save_dict, latest_path)

        if is_best:
            best_path = self.run_dir / "checkpoint_best.pt"
            torch.save(save_dict, best_path)

        self._checkpoint_log.append(
            {
                "step": step,
                "path": str(ckpt_path),
                "metrics": metrics,
                "is_best": is_best,
            }
        )

        # Prune old checkpoints (keep latest N + best)
        self._prune_old_checkpoints()

        return str(ckpt_path)

    def _prune_old_checkpoints(self):
        checkpoints = sorted(
            self.run_dir.glob("checkpoint_step*.pt"),
            key=lambda p: int(p.stem.replace("checkpoint_step", "")),
        )
        if len(checkpoints) > self.keep_n:
            for old in checkpoints[: -self.keep_n]:
                old.unlink(missing_ok=True)

    def save_safety_report(self, report: dict):
        """Save safety evaluation report to run dir."""
        out = self.run_dir / "safety_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

    def save_eval_results(self, results: dict):
        """Save evaluation results to run dir."""
        out = self.run_dir / "eval.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    def save_dataset_info(self, dataset_hash: str, dataset_path: str):
        """Record dataset provenance."""
        info = {
            "hash": dataset_hash,
            "path": dataset_path,
            "timestamp": datetime.now().isoformat(),
        }
        out = self.run_dir / "dataset_info.json"
        with open(out, "w") as f:
            json.dump(info, f, indent=2)

    def get_latest_checkpoint(self) -> str | None:
        """Return path to latest checkpoint or None."""
        latest = self.run_dir / "checkpoint.pt"
        return str(latest) if latest.exists() else None

    def load_for_resume(self, algorithm):
        """Load latest checkpoint into algorithm for training resume."""
        path = self.get_latest_checkpoint()
        if path is None:
            return 0  # start from scratch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "qf" in ckpt and hasattr(algorithm, "qf") and ckpt["qf"] is not None:
            algorithm.qf.load_state_dict(ckpt["qf"])
        if "policy" in ckpt and hasattr(algorithm, "policy"):
            algorithm.policy.load_state_dict(ckpt["policy"])
        if "actor" in ckpt and hasattr(algorithm, "actor"):
            algorithm.actor.load_state_dict(ckpt["actor"])
        if "model" in ckpt and hasattr(algorithm, "model"):
            algorithm.model.load_state_dict(ckpt["model"])
        return ckpt.get("step", 0)
