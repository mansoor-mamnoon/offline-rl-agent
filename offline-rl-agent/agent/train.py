import numpy as np
from cql import CQLAgent
from tqdm import trange




data = np.load("dataset/replay_buffer.npz")



agent = CQLAgent(state_dim=4, action_dim=3)
  # vector obs: (2,), 3 discrete actions

loss_history = {
    "bellman": [],
    "conservative": [],
    "bc": []
}


for epoch in trange(1000):
    idx = np.random.choice(len(data["actions"]), size=64)
    batch = {k: v[idx] for k, v in data.items()}

    losses = agent.train(batch)

    loss_history["bellman"].append(losses["bellman_loss"])
    loss_history["conservative"].append(losses["conservative_loss"])
    loss_history["bc"].append(losses["bc_loss"])


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(loss_history["bellman"], label="Bellman Loss")
plt.plot(loss_history["conservative"], label="Conservative Loss")
plt.plot(loss_history["bc"], label="Behavior Cloning Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CQL Training Losses Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("docs/cql_training_losses.png")  # Save the plot


if epoch % 100 == 0:
    print(f"Epoch {epoch} | Bellman: {losses['bellman_loss']:.3f} | Conservative: {losses['conservative_loss']:.3f} | BC: {losses['bc_loss']:.3f}")
