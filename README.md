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

---

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

---

## 🧠 Replay Buffer Generation

We simulate scripted agents in the custom Gridworld environment to collect experience data for offline RL training.

Each transition includes:
- `observation`
- `action`
- `reward`
- `next_observation`
- `done`

These transitions are saved into a compressed `.npz` buffer (`dataset/replay_buffer.npz`) for later use.

To generate the dataset:

```bash
python dataset/collect.py --episodes 100
```

Outputs:
- ✅ `dataset/metadata.txt` – Summary of average reward, length, and transitions
- 📊 `dataset/reward_histogram.png` – Reward distribution histogram

Example stats:
```
Num Episodes: 100
Avg Episode Reward: 8.30
Avg Episode Length: 18.00
Total Transitions: 1800
```

---

## 📊 Dataset Visualizations

We visualize the replay buffer to verify coverage and distribution:

- 🌀 t-SNE of Observations: `docs/plots/tsne_obs.png`
- 🎮 Action Distribution: `docs/plots/action_distribution.png`
- 🎯 Reward Distribution: `docs/plots/episode_rewards.png`

Generate plots via:

```bash
python dataset/viz.py
```

---

## 🏋️ Training the CQL Agent

We implement a Conservative Q-Learning (CQL) agent using PyTorch. The agent is trained *offline* on the replay buffer.

Key Features:
- Vector observation space (4D: [agent_x, agent_y, goal_x, goal_y])
- Discrete action space with 3 actions
- Bellman loss, conservative loss, and optional behavior cloning (BC) loss

Training includes:
- ✅ Evaluation loop (avg Q, policy accuracy)
- ✅ TensorBoard + Matplotlib logging
- ✅ Best checkpoint saving (`checkpoints/`)

Run:

```bash
python agent/train.py
```

---

## 📈 Training Visualizations

### 📉 Loss Curves (`docs/cql_training_losses.png`)

- **Bellman Loss**: TD error between predicted and target Q-values  
- **Conservative Loss**: Penalizes overestimation of unseen actions  
- **BC Loss**: Aligns policy with dataset behavior

![CQL Training Losses](docs/cql_training_losses.png)

### 🧪 TensorBoard Logging

All metrics are tracked in TensorBoard:

```bash
tensorboard --logdir=logs
```

Access at: [http://localhost:6006](http://localhost:6006)

Track:
- [`Eval/PolicyAccuracy`](http://localhost:6006/#scalars&tagFilter=PolicyAccuracy)
- [`Eval/AvgQ`](http://localhost:6006/#scalars&tagFilter=AvgQ)
- [`Loss/BC`](http://localhost:6006/#scalars&tagFilter=Loss%2FBC)
- [`Loss/Bellman`](http://localhost:6006/#scalars&tagFilter=Loss%2FBellman)

---

## 🎮 Agent Evaluation and Replay

After training, test the trained agent in the environment and save replay GIFs.

Run:

```bash
python agent/test_agent.py
```

Outputs:
- ✅ Printed reward over 10 episodes
- ✅ Replay saved as GIF: `docs/replays/test_run.gif`

Preview:

![Sample Replay](docs/replays/test_run.gif)

---

## 📁 Project Structure

```bash
offline-rl-agent/
│
├── env/
│   └── neuroquant_env.py         # Custom gridworld environment
│
├── dataset/
│   ├── collect.py                # Data generation script
│   ├── replay_buffer.npz         # Collected offline transitions
│   ├── metadata.txt              # Episode stats
│   ├── reward_histogram.png      # Reward histogram
│   ├── viz.py                    # t-SNE, action, reward plots
│
├── docs/
│   ├── cql_training_losses.png   # Training curves
│   └── replays/
│       └── test_run.gif          # GIF of trained agent behavior
│
├── agent/
│   ├── cql.py                    # CQL agent logic
│   ├── train.py                  # Training script
│   └── test_agent.py             # Inference and replay
│
├── checkpoints/                  # Saved model weights
│   ├── best_q.pt
│   └── best_policy.pt
│
└── logs/                         # TensorBoard logs
```

---

## ✅ Summary

| Component | Description |
|----------|-------------|
| Env | Custom 10x10 gridworld, image/vector obs |
| Dataset | 100-episode buffer with 1800 transitions |
| Agent | CQL agent with offline training |
| Logging | TensorBoard + Matplotlib |
| Inference | Replay and metrics saved |
| Visuals | t-SNE, reward histogram, action dist |

---

## 🧪 Try It Out

```bash
# Step-by-step
python dataset/collect.py --episodes 100
python dataset/viz.py
python agent/train.py
python agent/test_agent.py
```

---

## 📬 Contact

Feel free to reach out or open an issue for any questions or ideas!

