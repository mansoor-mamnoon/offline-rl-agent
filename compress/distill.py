import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path so we can import from agent/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.models import BigMLP, SmallMLP
from agent.test_agent import evaluate_policy

# === Settings ===
state_dim = 4
action_dim = 3
temperature = 2.0
batch_size = 128
num_epochs = 20
lr = 1e-3

# === Load dataset ===
buffer = np.load("dataset/replay_buffer.npz")
states = torch.tensor(buffer["observations"], dtype=torch.float32)

# === Load pretrained BigMLP (teacher) ===
teacher = BigMLP(state_dim, action_dim)
teacher.load_state_dict(torch.load("checkpoints/big_mlp.pt"))
teacher.eval()

# === Generate soft targets (teacher softmax logits) ===
with torch.no_grad():
    logits = teacher(states)
    soft_targets = F.softmax(logits / temperature, dim=1)

# === Create distillation dataset ===
dataset = TensorDataset(states, soft_targets)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Initialize SmallMLP (student) ===
student = SmallMLP(state_dim, action_dim)
optimizer = torch.optim.Adam(student.parameters(), lr=lr)
criterion = nn.KLDivLoss(reduction='batchmean')

# === Distillation training loop ===
os.makedirs("logs", exist_ok=True)
log_file = open("logs/distill_loss.log", "w")

for epoch in range(num_epochs):
    student.train()
    total_loss = 0.0
    for x, target in loader:
        optimizer.zero_grad()
        student_logits = student(x)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        loss = criterion(student_log_probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[Epoch {epoch}] Distillation Loss: {avg_loss:.4f}")
    log_file.write(f"{epoch},{avg_loss:.6f}\n")

log_file.close()

# === Save the distilled student model ===
torch.save(student.state_dict(), "checkpoints/small_mlp_distilled.pt")
print("✅ Distilled SmallMLP saved to checkpoints/small_mlp_distilled.pt")

# === (Optional) Evaluate student ===
def student_policy(obs):
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = student(obs_tensor)
        return torch.argmax(logits, dim=1).item()

# Call evaluate_policy from test_agent.py
reward = evaluate_policy(student_policy, episodes=10, render=False)
print(f"🎯 Student Final Reward: {reward:.2f}")

# Save reward to log
with open("logs/student_reward.log", "w") as f:
    f.write(f"{reward:.4f}\n")


import matplotlib.pyplot as plt

# === Read logs ===
loss_log_path = "logs/distill_loss.log"
reward_log_path = "logs/student_reward.log"

# Load loss log
epochs = []
losses = []
with open(loss_log_path, "r") as f:
    for line in f:
        epoch, loss = line.strip().split(",")
        epochs.append(int(epoch))
        losses.append(float(loss))

# Load final reward
with open(reward_log_path, "r") as f:
    final_reward = float(f.read().strip())

# === Plot 1: Distillation Loss ===
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', label="Distillation Loss")
plt.title("Student MLP Distillation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("KL Divergence Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
os.makedirs("logs", exist_ok=True)
plt.savefig("logs/distill_loss_plot.png")
print("📈 Saved distill_loss_plot.png")

# === Plot 2: Final Reward Bar Plot ===
plt.figure(figsize=(6, 4))
plt.bar(["Distilled Student"], [final_reward], color="green")
plt.title("Final Reward of Distilled Student MLP")
plt.ylabel("Average Reward (Over 10 Episodes)")
plt.ylim(0, max(10, final_reward + 1))
plt.tight_layout()
plt.savefig("logs/student_reward_plot.png")
print("🎯 Saved student_reward_plot.png")


