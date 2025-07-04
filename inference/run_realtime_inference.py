import os
import sys
import time
import torch
import numpy as np
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.neuroquant_env import NeuroQuantEnv
from agent.models import SmallMLP

# === Load distilled model ===
model_path = "checkpoints/small_mlp_distilled.pt"
assert os.path.exists(model_path), f"Model not found at {model_path}"

model = SmallMLP(state_dim=4, action_dim=3)
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cpu")

# === Init environment ===
env = NeuroQuantEnv(obs_mode="vector")
obs, _ = env.reset()
done = False

# === Performance tracking ===
frame_times = []
frame_count = 0
start_time = time.time()

print("🎮 Starting real-time inference loop...")
while not done:
    frame_start = time.perf_counter()

    # Preprocess
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        action = model(obs_tensor).argmax(dim=1).item()

    # Environment step
    obs, reward, done, _, _ = env.step(action)

    # Timing
    frame_end = time.perf_counter()
    latency_ms = (frame_end - frame_start) * 1000
    frame_times.append(latency_ms)
    frame_count += 1

    # Logging
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    fps = 1000 / latency_ms if latency_ms > 0 else float("inf")
    print(f"[Frame {frame_count}] Latency: {latency_ms:.2f} ms | FPS: {fps:.2f} | Mem: {mem:.2f} MB | Action: {action}")

    # Real-time pacing (≤ 100ms per step)
    target_frame_time = 0.1
    time.sleep(max(0.0, target_frame_time - (frame_end - frame_start)))

# === Final stats ===
avg_latency = sum(frame_times) / len(frame_times)
fps = 1000 / avg_latency if avg_latency > 0 else float("inf")
total_time = time.time() - start_time

print("\n🎯 Real-Time Inference Complete")
print(f"🕒 Total Time: {total_time:.2f} s")
print(f"📈 Avg Latency: {avg_latency:.2f} ms | Avg FPS: {fps:.2f}")



import csv

# === Save latency/FPS log ===
os.makedirs("logs", exist_ok=True)
with open("logs/day11_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "latency_ms", "fps"])
    for i, lat in enumerate(frame_times):
        fps = 1000 / lat if lat > 0 else 0
        writer.writerow([i + 1, lat, fps])

print("📝 Saved metrics to logs/day11_metrics.csv")
