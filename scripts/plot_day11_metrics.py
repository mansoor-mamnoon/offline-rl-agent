import os
import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV ===
df = pd.read_csv("logs/day11_metrics.csv")

# === Plot Latency ===
plt.figure(figsize=(10, 5))
plt.plot(df["frame"], df["latency_ms"], marker="o", linewidth=1)
plt.title("Per-Frame Inference Latency (ms)")
plt.xlabel("Frame")
plt.ylabel("Latency (ms)")
plt.grid(True)
plt.tight_layout()
os.makedirs("docs/plots", exist_ok=True)
plt.savefig("docs/plots/day11_latency.png")
print("📈 Saved latency plot to docs/plots/day11_latency.png")

# === Plot FPS ===
plt.figure(figsize=(10, 5))
plt.plot(df["frame"], df["fps"], marker="s", color="green", linewidth=1)
plt.title("Per-Frame Inference FPS")
plt.xlabel("Frame")
plt.ylabel("Frames per Second")
plt.grid(True)
plt.tight_layout()
plt.savefig("docs/plots/day11_fps.png")
print("🎞️ Saved FPS plot to docs/plots/day11_fps.png")
