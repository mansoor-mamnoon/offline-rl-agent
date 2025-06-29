## 🚀 Project Overview

NeuroQuant Agent is a fully custom offline reinforcement learning benchmark, built from the ground up with real-time constraints, compression-aware inference, and deployment to latency-constrained environments.

The project begins with a custom-built 10×10 gridworld environment that supports:

- 🔁 **Directional movement**: The agent can turn left, go forward, or turn right relative to its current orientation.
- 👁️ **Partial observability**: Instead of seeing the entire map, the agent receives a 3×3 view centered around its position.
- ⛔ **Obstacles**: Impassable wall tiles block the agent's path and require navigation.
- 🎯 **Goal tile**: A single terminal state gives a large positive reward when reached, ending the episode.
- 🖥️ **Real-time PyGame rendering**: Each simulation step is rendered at a capped 10 FPS for visual inspection and timing fidelity.

This environment is used as the basis for:
- Generating offline replay buffers
- Training offline RL agents using CQL, BCQ, or TD3+BC
- Benchmarking model compression tradeoffs (quantization, pruning, distillation)
- Real-time deployment of agents under latency and memory constraints


## 🧠 Environment Design

The environment is a 10×10 gridworld with directional agent movement, obstacles, and a single terminal goal. Key features:

- 🔁 **Action space**: Turn left, move forward, turn right (relative to current orientation)
- 👁️ **Partial observability**: Agent receives a 3×3 window centered on its current location
- 🔢 **Dual observation modes**:
  - **Image**: 3×3 local grid (int matrix)
  - **Vector**: Agent position and goal coordinates as a flat vector
- 🎯 **Reward structure**:
  - `+10` for reaching the goal (sparse)
  - `-0.1` per step (dense penalty)
- ⛔ **Obstacles**: Defined in the grid and block movement
- 🖥️ **Real-time rendering**: PyGame visualization at 10 FPS


## 🎥 Demos + GIFs

The environment supports saving full episodes as GIFs using the `render_episode_gif()` function.

Sample run saved to `docs/replays/test_run.gif`:
![Sample Replay](docs/replays/test_run.gif)
