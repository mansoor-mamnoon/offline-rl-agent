import numpy as np
from cql import CQLAgent
from tqdm import trange




data = np.load("dataset/replay_buffer.npz")



agent = CQLAgent(state_dim=4, action_dim=3)
  # vector obs: (2,), 3 discrete actions

for epoch in trange(1000):
    idx = np.random.choice(len(data["actions"]), size=64)
    batch = {k: v[idx] for k, v in data.items()}

    losses = agent.train(batch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Bellman: {losses['bellman_loss']:.3f} | Conservative: {losses['conservative_loss']:.3f} | BC: {losses['bc_loss']:.3f}")
