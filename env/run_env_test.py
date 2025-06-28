from neuroquant_env import NeuroQuantEnv
import time

env = NeuroQuantEnv()
obs, _ = env.reset()

for _ in range(30):
    action = env.action_space.sample()
    obs, reward, done, trunc, _ = env.step(action)
    env.render()
    time.sleep(0.1)  # slows down the loop so you can see it

env.close()

# Prevent the script from auto-closing immediately
input("Press Enter to close the window...")
