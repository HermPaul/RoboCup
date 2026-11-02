import time
from ssl_strategy_api.rl.striker_env import StrikerEnv

env = StrikerEnv(seed=0)
obs, _ = env.reset()

t0 = time.perf_counter()
n, episodes, ep_len, lengths = 0, 0, 0, []
for _ in range(2000):
    obs, r, done, trunc, _ = env.step(env.action_space.sample())
    n += 1
    ep_len += 1
    if done or trunc:
        lengths.append(ep_len)
        ep_len = 0
        episodes += 1
        env.reset()

dt = time.perf_counter() - t0
print(f"{n / dt:.0f} steps/s, {episodes} episodes, lengths: {lengths[:15]}")
