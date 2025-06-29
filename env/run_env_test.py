from neuroquant_env import NeuroQuantEnv
import time
import os

# === Setup ===
env = NeuroQuantEnv()
env.obs_mode = "vector"  # Change to "image" to test 3x3 partial grid

# === Rollout ===
obs, _ = env.reset()
total_reward = 0

print(f"Initial observation (mode: {env.obs_mode}):\n{obs}")

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, trunc, _ = env.step(action)
    total_reward += reward
    print(f"Step {step:03d} | Action: {action} | Reward: {reward:.2f} | Obs:\n{obs}")
    env.render()
    time.sleep(0.2)
    if done:
        print("Reached goal!")
        break

print(f"\nTotal episode reward: {total_reward:.2f}")

# === Save Replay ===
os.makedirs("docs/replays", exist_ok=True)


env.render_episode_gif(path="docs/replays/test_run.gif")
env.close()

input("Press Enter to close...")
