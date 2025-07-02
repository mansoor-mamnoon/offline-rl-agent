import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import os
from cql import CQLAgent

# Load dataset
data = np.load("dataset/replay_buffer.npz")

# Init agent
agent = CQLAgent(state_dim=4, action_dim=3)

# Logging setup
logdir = f"logs/cql_{int(time.time())}"
writer = SummaryWriter(logdir)

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

loss_history = {
    "bellman": [],
    "conservative": [],
    "bc": []
}


def evaluate(agent, data, num_samples=1000):
    """Evaluate average predicted Q-value and policy accuracy."""
    s = torch.tensor(data["observations"][:num_samples], dtype=torch.float32)
    a = torch.tensor(data["actions"][:num_samples], dtype=torch.long)

    with torch.no_grad():
        logits = agent.policy(s)
        pred_actions = torch.argmax(logits, dim=1)
        policy_acc = (pred_actions == a).float().mean().item()

        a_onehot = F.one_hot(a, agent.action_dim).float()
        q_vals = agent.q_net(s, a_onehot)
        avg_q = q_vals.mean().item()

    return policy_acc, avg_q


# Training loop
best_acc = 0.0

for epoch in trange(1000):
    idx = np.random.choice(len(data["actions"]), size=64)
    batch = {k: v[idx] for k, v in data.items()}
    losses = agent.train(batch)

    # Track training loss
    loss_history["bellman"].append(losses["bellman_loss"])
    loss_history["conservative"].append(losses["conservative_loss"])
    loss_history["bc"].append(losses["bc_loss"])

    # Log training losses
    writer.add_scalar("Loss/Bellman", losses['bellman_loss'], epoch)
    writer.add_scalar("Loss/Conservative", losses['conservative_loss'], epoch)
    writer.add_scalar("Loss/BC", losses['bc_loss'], epoch)

    # Evaluate every 100 steps
    if epoch % 100 == 0:
        acc, avg_q = evaluate(agent, data)
        writer.add_scalar("Eval/PolicyAccuracy", acc, epoch)
        writer.add_scalar("Eval/AvgQ", avg_q, epoch)

        print(f"[Eval @ {epoch}] Acc: {acc:.3f} | Avg Q: {avg_q:.3f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(agent.q_net.state_dict(), "checkpoints/best_q.pt")
            torch.save(agent.policy.state_dict(), "checkpoints/best_policy.pt")

# Plot and save training curves after training
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
plt.savefig("docs/cql_training_losses.png")
