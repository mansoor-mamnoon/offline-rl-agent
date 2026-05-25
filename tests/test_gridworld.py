import os
import tempfile
import numpy as np
import pytest

from offline_rl.envs.gridworld import GridWorld, GridWorldConfig, random_policy, near_optimal_policy
from offline_rl.envs.cliffwalking import make_cliff_walking


def make_env(state_repr="normalized"):
    config = GridWorldConfig(grid_size=10, start=(0, 0), goal=(9, 9), max_steps=200, state_repr=state_repr)
    return GridWorld(config)


def test_reset_correct_state():
    env = make_env()
    obs = env.reset()
    assert obs.shape == (2,)
    assert np.allclose(obs, [0.0, 0.0])


def test_reset_onehot_state():
    env = make_env(state_repr="onehot")
    obs = env.reset()
    assert obs.shape == (100,)
    assert obs[0] == 1.0
    assert obs.sum() == 1.0


def test_episode_terminates_at_goal():
    config = GridWorldConfig(grid_size=3, start=(0, 0), goal=(2, 2), max_steps=50)
    env = GridWorld(config)
    env.reset()
    # Navigate to goal manually: down, down, right, right
    moves = [1, 1, 3, 3]
    done = False
    for a in moves:
        obs, reward, done, info = env.step(a)
    assert done
    assert reward == 10.0
    assert info["reached_goal"]


def test_episode_terminates_at_cliff():
    config = GridWorldConfig(
        grid_size=5,
        start=(0, 0),
        goal=(4, 4),
        cliff_cells=[(1, 0)],
        max_steps=50,
    )
    env = GridWorld(config)
    env.reset()
    obs, reward, done, info = env.step(1)  # move down into cliff
    assert done
    assert reward == -10.0
    assert info["fell_off_cliff"]


def test_max_steps_terminates():
    config = GridWorldConfig(grid_size=5, start=(0, 0), goal=(4, 4), max_steps=3)
    env = GridWorld(config)
    env.reset()
    for _ in range(3):
        obs, reward, done, info = env.step(0)  # keep moving up (hits boundary)
    assert done


def test_dataset_generation_shapes():
    env = make_env()
    dataset = env.generate_dataset(random_policy, n_episodes=10)
    n = len(dataset["observations"])
    assert n > 0
    assert dataset["observations"].shape == (n, 2)
    assert dataset["actions"].shape == (n,)
    assert dataset["rewards"].shape == (n,)
    assert dataset["next_observations"].shape == (n, 2)
    assert dataset["dones"].shape == (n,)
    assert dataset["episode_ids"].shape == (n,)
    # Episode IDs range from 0 to 9
    assert dataset["episode_ids"].max() == 9
    assert dataset["episode_ids"].min() == 0


def test_hdf5_save_load_roundtrip():
    import h5py

    env = make_env()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_dataset.h5")
        dataset = env.generate_dataset(random_policy, n_episodes=5, save_path=path)
        assert os.path.exists(path)
        with h5py.File(path, "r") as f:
            loaded_obs = f["observations"][:]
        np.testing.assert_array_equal(loaded_obs, dataset["observations"])


def test_cliff_walking_state_shape():
    env = make_cliff_walking()
    obs = env.reset()
    assert obs.shape == (2,)


def test_obstacle_prevents_movement():
    config = GridWorldConfig(
        grid_size=5,
        start=(0, 0),
        goal=(4, 4),
        obstacle_cells=[(0, 1)],
        max_steps=10,
    )
    env = GridWorld(config)
    env.reset()
    obs, _, _, info = env.step(3)  # move right — should be blocked
    assert info["pos"] == (0, 0)
