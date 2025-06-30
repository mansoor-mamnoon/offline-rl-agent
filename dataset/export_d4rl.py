import h5py
import numpy as np

buffer = np.load("dataset/replay_buffer.npz")

with h5py.File("dataset/replay_d4rl.hdf5", "w") as f:
    f.create_dataset("observations", data=buffer["observations"])
    f.create_dataset("actions", data=buffer["actions"])
    f.create_dataset("rewards", data=buffer["rewards"])
    f.create_dataset("terminals", data=buffer["dones"])
    f.create_dataset("next_observations", data=buffer["next_observations"])


