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

## 🧠 Replay Buffer Generation

We simulate random or scripted agents in the custom Gridworld environment to collect experience data for offline RL training.

Each transition includes:
- `observation`
- `action`
- `reward`
- `next_observation`
- `done`

These transitions are saved into a compressed `.npz` buffer (`dataset/replay_buffer.npz`), which can later be loaded for training Conservative Q-Learning (CQL), TD3+BC, or BCQ agents.

Additional outputs include:
- ✅ Episode metadata (average reward, length, and total transitions) saved to `dataset/metadata.txt`
- 📊 A histogram of reward distribution over episodes saved to `dataset/reward_histogram.png`

To generate the dataset, run:

```bash
python dataset/collect.py --episodes 100
```

This will generate 10k+ transitions across 100 episodes using a random policy.

---

## 📁 Project Structure

```bash
offline-rl-agent/
│
├── env/                    # Custom Gym environment (NeuroQuantEnv)
│   └── neuroquant_env.py
│
├── dataset/                # Replay buffer collection + visualizations
│   ├── collect.py          # Random/scripted policy buffer generation
│   ├── viz.py              # t-SNE, reward, and action plots
│   ├── replay_buffer.npz   # (gitignored) Collected transitions
│   ├── reward_histogram.png
│   ├── metadata.txt
│
├── docs/
│   └── plots/              # Visual outputs of dataset
│       ├── tsne_obs.png
│       ├── action_distribution.png
│       └── episode_rewards.png
│
├── .gitignore
├── README.md
└── run_env_test.py         # Debug script to manually interact with env
```

---

## 📊 Dataset Visualizations

We visualize the replay buffer to verify coverage and distribution:

- 🌀 [t-SNE of Observations](docs/plots/tsne_obs.png): clusters state embeddings in 2D
- 🎮 [Action Distribution](docs/plots/action_distribution.png): histogram over agent actions
- 🎯 [Episode Reward Distribution](docs/plots/episode_rewards.png): how returns are spread across episodes

These plots are generated via:

```bash
python dataset/viz.py
```

---

## 📦 Getting Started

```bash
# 1. Clone and enter the repo
git clone https://github.com/mansoor-mamnoon/offline-rl-agent.git
cd offline-rl-agent

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run environment manually
python env/run_env_test.py

# 5. Collect dataset
python dataset/collect.py --episodes 100

# 6. Visualize dataset
python dataset/viz.py
```

---





## 🎥 Demos + GIFs

The environment supports saving full episodes as GIFs using the `render_episode_gif()` function.

Sample run saved to `docs/replays/test_run.gif`:
![Sample Replay](docs/replays/test_run.gif)


