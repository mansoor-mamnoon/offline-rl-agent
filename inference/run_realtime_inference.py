import os
import sys
import time
import torch
import numpy as np
import psutil
import datetime
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.neuroquant_env import NeuroQuantEnv
from agent.models import SmallMLP

# === Session file logic ===
os.makedirs("results", exist_ok=True)
existing = [f for f in os.listdir("results") if f.startswith("session_") and f.endswith(".csv")]
session_ids = [int(f.split("_")[1].split(".")[0]) for f in existing if f.split("_")[1].split(".")[0].isdigit()]
session_id = max(session_ids) + 1 if session_ids else 1
session_file = f"results/session_{session_id}.csv"

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

# === Tracking setup ===
frame_times = []
frame_count = 0
total_reward = 0
frame_log = []
start_time = time.time()

print("🎮 Starting real-time inference loop...")

while not done:
    frame_start = time.perf_counter()

    # Inference
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        action = model(obs_tensor).argmax(dim=1).item()

    # Step env
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    # Timing and metrics
    frame_end = time.perf_counter()
    latency_ms = (frame_end - frame_start) * 1000
    frame_times.append(latency_ms)
    frame_count += 1
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    fps = 1000 / latency_ms if latency_ms > 0 else float("inf")
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')

    print(f"[Frame {frame_count}] Latency: {latency_ms:.2f} ms | FPS: {fps:.2f} | Mem: {mem:.2f} MB | Reward: {reward:.2f} | Action: {action}")
    frame_log.append([timestamp, round(latency_ms, 2), round(mem, 2), round(total_reward, 2)])

    # Auto-shutdown on high latency
    if latency_ms > 150:
        print(f"⛔ Latency {latency_ms:.2f}ms exceeded threshold. Auto-shutdown at frame {frame_count}.")
        break

    # Real-time pacing (≤100ms per frame)
    target_frame_time = 0.1
    time.sleep(max(0.0, target_frame_time - (frame_end - frame_start)))

# === Save CSV ===
with open(session_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "latency_ms", "memory_mb", "cumulative_reward"])
    writer.writerows(frame_log)

print(f"📝 Session log saved to {session_file}")

# === Final stats ===
avg_latency = sum(frame_times) / len(frame_times)
fps = 1000 / avg_latency if avg_latency > 0 else float("inf")
total_time = time.time() - start_time

print("\n🎯 Real-Time Inference Complete")
print(f"🕒 Total Time: {total_time:.2f} s")
print(f"📈 Avg Latency: {avg_latency:.2f} ms | Avg FPS: {fps:.2f}")
