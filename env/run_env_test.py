from neuroquant_env import NeuroQuantEnv
import time

env = NeuroQuantEnv()
obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, trunc, _ = env.step(action)
    env.render()
    time.sleep(1)
    if done:
        break

env.close()
input("Press Enter to close...")
