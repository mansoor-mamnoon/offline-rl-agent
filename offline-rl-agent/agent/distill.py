import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent.models import BigMLP, SmallMLP

# Settings
state_dim = 4
action_dim = 3
temperature = 2.0
batch_size = 128
num_epochs = 20
lr = 1e-3

# Load dataset
data = np.load("dataset/replay_buffer.npz")
X = torch.tensor(data["observations"], dtype=torch.float32)

# Load teacher (BigMLP)
teacher = BigMLP(state_dim, action_dim)
teacher.load_state_dict(torch.load("checkpoints/big_mlp.pt"))
teacher.eval()

# Generate soft targets
with torch.no_grad():
    logits = teacher(X)
    soft_targets = F.softmax(logits / temperature, dim=1)

dataset = TensorDataset(X, soft_targets)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Init student (SmallMLP)
student = SmallMLP(state_dim, action_dim)
optimizer = torch.optim.Adam(student.parameters(), lr=lr)
loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for xb, yb in loader:
        logits = student(xb)
        log_probs = F.log_softmax(logits / temperature, dim=1)
        loss = loss_fn(log_probs, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch}] Distill Loss: {total_loss / len(loader):.4f}")

# Save student
torch.save(student.state_dict(), "checkpoints/small_mlp_distilled.pt")
print("✅ Saved distilled SmallMLP to checkpoints/small_mlp_distilled.pt")
