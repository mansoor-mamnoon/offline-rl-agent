import os
import pandas as pd
import matplotlib.pyplot as plt

# === Locate latest session file ===
results_dir = "results"
session_files = sorted([f for f in os.listdir(results_dir) if f.startswith("session_") and f.endswith(".csv")])
latest_file = os.path.join(results_dir, session_files[-1])
print(f"📄 Plotting latest session: {latest_file}")

# === Load data ===
df = pd.read_csv(latest_file)

# === Plot 1: Latency over time ===
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["latency_ms"], marker="o", linewidth=1)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Timestamp")
plt.ylabel("Latency (ms)")
plt.title("Per-Frame Latency Over Time")
plt.grid(True)
plt.tight_layout()
os.makedirs("docs/plots", exist_ok=True)
plt.savefig("docs/plots/day12_latency.png")
print("📈 Saved latency plot to docs/plots/day12_latency.png")

# === Plot 2: Cumulative Reward over time ===
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["cumulative_reward"], marker="s", color="green", linewidth=1)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Timestamp")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("docs/plots/day12_reward.png")
print("🎯 Saved reward plot to docs/plots/day12_reward.png")
