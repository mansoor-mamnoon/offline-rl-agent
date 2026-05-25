import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent.models import SmallMLP

# Settings
state_dim = 4
action_dim = 3
batch_size = 128
num_epochs = 20
lr = 1e-3

# Load dataset
data = np.load("dataset/replay_buffer.npz")
X = torch.tensor(data["observations"], dtype=torch.float32)
y = torch.tensor(data["actions"], dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Init model
model = SmallMLP(state_dim, action_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch}] Loss: {total_loss / len(loader):.4f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/small_mlp.pt")
print("✅ Saved SmallMLP to checkpoints/small_mlp.pt")
