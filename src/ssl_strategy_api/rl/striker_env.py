import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from ssl_strategy_api.run_demos.run_rsoccer_exercise1 import Exercise1Env
from ssl_strategy_api.adapters.rsoccer import to_rsoccer_commands
from ssl_strategy_api.models import RobotIntent
from ssl_strategy_api.rl.common import build_obs, action_to_intent, OBS_DIM


class StrikerEnv(gym.Env):
    def __init__(self, seed=None, max_steps=600):    # 600 * 0.025 s = 15 s
        super().__init__()
        self.sim = Exercise1Env(spawner_seed=seed)
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state, _ = self.sim.reset(seed=seed)
        self.t = 0
        # TODO: store whatever your reward needs from the previous step

        return build_obs(self.state, self.sim.blue_config), {}


    def step(self, action):
        heading = self.state.robots_blue[0].heading_rad or 0.0
        intent = action_to_intent(action, heading)
        cmds = to_rsoccer_commands(self.state, self.sim.blue_config, {0: intent})
        cmds.extend(to_rsoccer_commands(self.state, self.sim.yellow_config, {0: RobotIntent()}))
        self.state, _, done, _, _ = self.sim.step(cmds)
        self.t += 1

        reward = self.compute_reward(self.state, done)
        truncated = self.t >= self.max_steps
        return build_obs(self.state, self.sim.blue_config), reward, bool(done), truncated, {}


    def compute_reward(self, state, done):
        return 0.0   # TODO: your job, see hints below
    