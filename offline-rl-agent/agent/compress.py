# agent/compress.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.utils.prune as prune
import time
import psutil
import matplotlib.pyplot as pl
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from env.neuroquant_env import NeuroQuantEnv
from agent.models import PolicyNetwork
import time
torch.backends.quantized.engine = 'qnnpack'
torch_device = torch.device('cpu')  # Force CPU

def finetune_pruned(model, episodes=30):
    print(f"🎯 Starting finetuning over {episodes} episodes...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    env = NeuroQuantEnv(obs_mode="vector")
    start = time.time()
    for ep in range(episodes):
        print(f"🔁 Episode {ep + 1}/{episodes}")
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 300:  # prevent infinite loops
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Forward pass
            logits = model(obs_tensor)

            # Use the *original model's* policy as the target (imitate itself)
            with torch.no_grad():
                target_action = policy(obs_tensor).argmax(dim=1)

            # Behavior cloning loss: imitate the original model’s decisions
            loss = F.cross_entropy(logits, target_action)

            # Backprop and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Take actual environment step
            action = logits.argmax(dim=1).item()
            obs, reward, done, _, _ = env.step(action)
            steps += 1

# Ensure proper quantization engine
torch.backends.quantized.engine = 'qnnpack'

# Load original model
policy = PolicyNetwork(state_dim=4, action_dim=3)
policy.load_state_dict(torch.load("checkpoints/best_policy.pt"))
policy.eval()

# ---------------- Static Quantization ----------------
# Step 1: Create float model for quantization
quantized_model = PolicyNetwork(state_dim=4, action_dim=3)
quantized_model.load_state_dict(torch.load("checkpoints/best_policy.pt"))
quantized_model.eval()

# Step 2: Set quant config
quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Step 3: Prepare for quantization (adds observers)
torch.quantization.prepare(quantized_model, inplace=True)

# Step 4: Calibrate with dummy env states
with torch.no_grad():
    for _ in range(100):
        inputs = torch.randn(32, 4)
        quantized_model(inputs)

# ✅ Step 5: Finetune BEFORE conversion
print("🎯 Finetuning quantized model...")
finetune_pruned(quantized_model, episodes=100)

# ✅ Step 6: Now convert (in-place) — params become non-trainable
torch.quantization.convert(quantized_model, inplace=True)

# ✅ Step 7: Save the quantized model
torch.save(quantized_model.state_dict(), "checkpoints/quantized_static.pt")




# ---------------- Pruning ----------------
# pruned_model = PolicyNetwork(state_dim=4, action_dim=3)
# pruned_model.load_state_dict(torch.load("checkpoints/best_policy.pt"))
# pruned_model.eval()

# ✅ Unstructured pruning on last linear layer
# for name, module in reversed(list(pruned_model.named_modules())):
#     if isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name='weight', amount=0.1)
#         print(f"✅ Pruned final linear layer: {name}")
#         break

# ✅ Structured pruning on hidden layer with 256 outputs
# ✅ Structured pruning on hidden layer with 256 outputs
# ---------------- Pruning ----------------
pruned_model = PolicyNetwork(state_dim=4, action_dim=3)
pruned_model.load_state_dict(torch.load("checkpoints/best_policy.pt"))
pruned_model.eval()

# Structured pruning: remove 30% of hidden channels in all hidden layers
for name, module in pruned_model.named_modules():
    if isinstance(module, torch.nn.Linear) and module.out_features == 512:
        prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
        print(f"✅ Structured-pruned layer: {name}")






    print("✅ Finetuning completed.")





print("🎯 Finetuning pruned model...")
finetune_pruned(pruned_model, episodes=100)



# Remove pruning hooks after finetuning
for name, module in pruned_model.named_modules():
    if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight_orig'):
        prune.remove(module, 'weight')



# ---------------- Evaluation ----------------
from time import perf_counter

def evaluate(model, label):
    env = NeuroQuantEnv(obs_mode="vector")
    total_reward, total_steps = 0, 0
    latencies = []

    obs, _ = env.reset()
    done = False

    while not done and total_steps < 200:
        # Use batch inference to simulate realistic latency
        obs_batch = np.stack([obs for _ in range(32)], axis=0)
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)

        start = perf_counter()
        with torch.no_grad():
            actions = model(obs_tensor).argmax(dim=1)
            action = actions[0].item()
        latencies.append(perf_counter() - start)

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1

    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    avg_latency_ms = 1000 * sum(latencies) / len(latencies)

    print(f"📦 [{label}] Reward: {total_reward:.2f} | Latency: {avg_latency_ms:.2f}ms | Mem: {mem:.2f} MB")
    return total_reward, avg_latency_ms, mem

# Run evaluations
quantized_model.to(torch_device)

r1, l1, m1 = evaluate(policy, "Original")
# r2, l2, m2 = evaluate(quantized_model, "Quantized")
r3, l3, m3 = evaluate(pruned_model, "Pruned")

# ---------------- Visualization ----------------
models = ["Original", "Pruned"]
rewards = [r1,r3]
latencies = [l1, l3]
mems = [m1, m3]

plt.figure(figsize=(10, 5))
scatter = plt.scatter(latencies, rewards, s=200, c=mems, cmap="cool", edgecolors="k")
for i, label in enumerate(models):
    plt.text(latencies[i]+0.1, rewards[i], label)

plt.xlabel("Latency (ms)")
plt.ylabel("Reward")
plt.title("Reward vs Latency (Bubble = Memory)")
cbar = plt.colorbar(scatter)
cbar.set_label("Memory (MB)")
plt.tight_layout()
plt.savefig("docs/plots/compression_tradeoff.png")
plt.close()
