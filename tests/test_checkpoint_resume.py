"""Tests for checkpoint saving and loading."""
import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile


def make_bc(obs_dim=8, act_dim=2):
    from offline_rl.algorithms.bc import BehaviorCloning, BCConfig

    return BehaviorCloning(obs_dim, act_dim, BCConfig(), device="cpu")


def test_save_and_load_latest(tmp_path):
    from offline_rl.training.checkpointing import CheckpointManager

    algo = make_bc()
    cm = CheckpointManager(tmp_path)
    path = cm.save(algo, step=100, metrics={"loss": 0.5})
    assert Path(path).exists()
    assert (tmp_path / "checkpoint.pt").exists()


def test_load_for_resume(tmp_path):
    from offline_rl.training.checkpointing import CheckpointManager

    algo = make_bc()
    cm = CheckpointManager(tmp_path)
    cm.save(algo, step=200, metrics={"loss": 0.3})
    algo2 = make_bc()
    step = cm.load_for_resume(algo2)
    assert step == 200


def test_dataset_hash_deterministic():
    from offline_rl.training.checkpointing import compute_dataset_hash

    dataset = {
        "observations": np.zeros((100, 8), dtype=np.float32),
        "actions": np.ones((100, 2), dtype=np.float32),
    }
    h1 = compute_dataset_hash(dataset)
    h2 = compute_dataset_hash(dataset)
    assert h1 == h2
    assert len(h1) == 12


def test_save_safety_report(tmp_path):
    from offline_rl.training.checkpointing import CheckpointManager

    cm = CheckpointManager(tmp_path)
    cm.save_safety_report({"mean_return": 42.0, "slo_violation_rate": 0.05})
    assert (tmp_path / "safety_report.json").exists()


def test_prune_old_checkpoints(tmp_path):
    from offline_rl.training.checkpointing import CheckpointManager

    algo = make_bc()
    cm = CheckpointManager(tmp_path, keep_n=2)
    for step in [100, 200, 300, 400]:
        cm.save(algo, step=step, metrics={"loss": 0.5})
    remaining = list(tmp_path.glob("checkpoint_step*.pt"))
    assert len(remaining) <= 2


def test_get_latest_checkpoint_none(tmp_path):
    from offline_rl.training.checkpointing import CheckpointManager

    cm = CheckpointManager(tmp_path)
    assert cm.get_latest_checkpoint() is None


def test_save_eval_results(tmp_path):
    from offline_rl.training.checkpointing import CheckpointManager

    cm = CheckpointManager(tmp_path)
    cm.save_eval_results({"mean_return": 55.0, "std_return": 3.0})
    assert (tmp_path / "eval.json").exists()


def test_save_dataset_info(tmp_path):
    from offline_rl.training.checkpointing import CheckpointManager

    cm = CheckpointManager(tmp_path)
    cm.save_dataset_info("abc123", "/data/traffic.h5")
    assert (tmp_path / "dataset_info.json").exists()
