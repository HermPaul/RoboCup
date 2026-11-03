import argparse
import csv
from importlib import import_module
from math import ceil
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

from ssl_strategy_api import FieldGeometry, TeamColor, TeamConfig
from ssl_strategy_api.adapters.rsoccer import (
    to_rsoccer_commands,
    world_state_from_rsoccer_frame,
)
from ssl_strategy_api.helpers import BallSpawner
from ssl_strategy_api.models import RobotIntent
from ssl_strategy_api.run_demos.reward_evaluators.touch_and_shoot_evaluator import TouchAndShootEvaluator
from ssl_strategy_api.run_demos.reward_evaluators.touch_evaluator import TouchEvaluator

SIMULATION_STEP_SECONDS = 0.025
DEFAULT_EPISODE_DURATION_SECONDS = 6.0
BLUE_ROBOT_ID = 0
YELLOW_ROBOT_ID = 0
RIGHT_SIDE_MARGIN_M = 0.20
BALL_IDLE_SPEED_MPS = 0.06
DEFAULT_NUM_EPISODES = 10000
SAVE_FREQ = 500
NUM_ENVS = 8
WEIGHTS_DIR = "ssl_strategy_api/src/ssl_strategy_api/strategies/weights/striker/"
WEIGHTS_DIR_THIS_RUN = f"{WEIGHTS_DIR}run_{time.strftime('%Y%m%d_%H%M%S')}/"
DEFAULT_LOG_FILENAME = f"ssl_strategy_api/logs/run_{time.strftime('%Y%m%d_%H%M%S')}.csv"
LOG_FREQ = 20


class TrainingEnv(SSLBaseEnv):
    """Headless environment configured for parallelized AsyncVectorEnv workers."""

    def __init__(
            self,
            spawner_seed: Optional[int] = None,
            evaluator: Optional[Any] = None,
    ):
        self.blue_config = TeamConfig(
            team=TeamColor.BLUE,
            controlled_robot_ids=(BLUE_ROBOT_ID,),
            attack_direction_x=1,
        )
        self.yellow_config = TeamConfig(
            team=TeamColor.YELLOW,
            controlled_robot_ids=(YELLOW_ROBOT_ID,),
            attack_direction_x=-1,
        )
        super().__init__(
            field_type=1,
            n_robots_blue=1,
            n_robots_yellow=1,
            time_step=SIMULATION_STEP_SECONDS,
            render_mode=None,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self.canonical_field = FieldGeometry(
            length_m=self.field.length,
            width_m=self.field.width,
            goal_width_m=self.field.goal_width,
        )
        self.state = None
        self.ball_spawner = BallSpawner(seed=spawner_seed)
        self.evaluator = evaluator or TouchEvaluator()
        self.latest_info: Dict[str, Any] = {}

    def _get_commands(self, action: Any):
        # 1. Direct pass of rSoccer RobotCommand objects
        if isinstance(action, list) and len(action) > 0 and hasattr(action[0], "yellow"):
            return action

        # 2. Vectorized 5D action array from RL policy [vx, vy, omega, kick, dribbler]
        if isinstance(action, (list, tuple, np.ndarray)):
            act_flat = np.asarray(action, dtype=np.float32).flatten()
            if len(act_flat) == 5:
                blue_intents = {
                    BLUE_ROBOT_ID: RobotIntent(
                        vx_mps=float(np.clip(act_flat[0], -2.0, 2.0)),
                        vy_mps=float(np.clip(act_flat[1], -2.0, 2.0)),
                        angular_velocity_rad_s=float(np.clip(act_flat[2], -3.0, 3.0)),
                        kick_speed_mps=float(np.clip(act_flat[3], 0.0, 5.0)),
                        dribbler=bool(act_flat[4] > 0.0),
                    )
                }
                yellow_intents = {YELLOW_ROBOT_ID: RobotIntent()}

                commands = to_rsoccer_commands(self.state, self.blue_config, blue_intents)
                commands.extend(to_rsoccer_commands(self.state, self.yellow_config, yellow_intents))
                return commands

        raise ValueError(f"Invalid action passed to _get_commands: {type(action)} -> {action}")

    def _frame_to_observations(self) -> np.ndarray:
        self.state = world_state_from_rsoccer_frame(
            self.frame,
            self.canonical_field,
            frame_number=self.steps,
        )

        striker = self.state.robots_blue.get(BLUE_ROBOT_ID)
        goalkeeper = self.state.robots_yellow.get(YELLOW_ROBOT_ID)
        ball = self.state.ball

        if striker is None or goalkeeper is None or ball is None:
            return np.zeros(15, dtype=np.float32)

        return np.array([
            striker.x_m, striker.y_m, striker.heading_rad, striker.vx_mps, striker.vy_mps,
            striker.angular_velocity_rad_s, float(striker.infrared or 0),
            goalkeeper.x_m, goalkeeper.y_m, goalkeeper.vx_mps, goalkeeper.vy_mps,
            ball.x_m, ball.y_m, ball.vx_mps, ball.vy_mps
        ], dtype=np.float32)

    def _calculate_reward_and_done(self):
        reward, done, info = self.evaluator.calculate_reward_and_done(self)
        self.latest_info = info
        return reward, done

    def _get_initial_positions_frame(self):
        frame = Frame()
        blue_spawn = self.ball_spawner.spawn_stationary_right_side(
            self.field,
            attack_direction_x=self.blue_config.attack_direction_x,
            right_side_margin_m=RIGHT_SIDE_MARGIN_M,
            enforce_outside_opponent_goal_area=True,
        )
        ball_spawn = self.ball_spawner.spawn_stationary_right_side(
            self.field,
            attack_direction_x=self.blue_config.attack_direction_x,
            right_side_margin_m=RIGHT_SIDE_MARGIN_M,
            enforce_outside_opponent_goal_area=True,
        )

        frame.ball = Ball(
            x=ball_spawn.x_m,
            y=ball_spawn.y_m,
            v_x=ball_spawn.vx_mps,
            v_y=ball_spawn.vy_mps,
        )

        frame.robots_blue[BLUE_ROBOT_ID] = Robot(
            yellow=False,
            id=BLUE_ROBOT_ID,
            x=blue_spawn.x_m,
            y=blue_spawn.y_m,
            theta=180.0,
        )

        half_length = self.field.length / 2
        half_width = self.field.width / 2
        frame.robots_yellow[YELLOW_ROBOT_ID] = Robot(
            yellow=True,
            id=YELLOW_ROBOT_ID,
            x=half_length - 0.10,
            y=-(half_width),
            theta=180.0,
        )
        return frame


def make_env_factory(seed: Optional[int], env_id: int, task: str):
    def _thunk():
        evaluator = TouchAndShootEvaluator() if task == "shoot" else TouchEvaluator()
        env_seed = (seed + env_id) if seed is not None else None
        return TrainingEnv(spawner_seed=env_seed, evaluator=evaluator)

    return _thunk


def _load_strategy(path: str, checkpoint_path: Optional[str] = None, num_envs: int = 1) -> Any:
    module_name, separator, class_name = path.partition(":")
    if not separator:
        raise ValueError("strategy must use module.path:ClassName format")
    strategy_class = getattr(import_module(module_name), class_name)

    try:
        return strategy_class(checkpoint_path=checkpoint_path, num_envs=num_envs)
    except TypeError:
        strategy = strategy_class()
        if checkpoint_path and hasattr(strategy, "load_checkpoint"):
            strategy.load_checkpoint(checkpoint_path)
        return strategy


def train_vectorized(
        blue_strategy: Any,
        yellow_strategy: Any,
        num_episodes: int,
        num_envs: int = NUM_ENVS,
        task: str = "shoot",
        max_duration_seconds: float = DEFAULT_EPISODE_DURATION_SECONDS,
        log_filename: str = DEFAULT_LOG_FILENAME,
        seed: Optional[int] = None,
) -> None:
    env_fns = [make_env_factory(seed, i, task) for i in range(num_envs)]
    vector_env = gym.vector.AsyncVectorEnv(env_fns)

    max_steps_per_episode = ceil(max_duration_seconds / SIMULATION_STEP_SECONDS)

    os.makedirs(os.path.dirname(log_filename) or ".", exist_ok=True)
    log_file = open(log_filename, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "episode", "total_reward", "steps", "duration_seconds", "finish_reason", "touched_ball"
    ])

    print(f"Parallel training running on {num_envs} CPU workers. Target: {num_episodes} episodes...")
    start_time = time.time()
    completed_episodes = 0

    try:
        while completed_episodes < num_episodes:
            obs_batch, _ = vector_env.reset(seed=seed)
            if hasattr(blue_strategy, "reset"):
                blue_strategy.reset(seed=seed)
            if hasattr(yellow_strategy, "reset"):
                yellow_strategy.reset(seed=seed)

            total_rewards = np.zeros(num_envs)
            episode_steps = 0

            for step in range(1, max_steps_per_episode + 1):
                episode_steps = step

                # Batched GPU forward pass yields (num_envs, 5) action array
                actions_batch = blue_strategy.decide_batch(obs_batch)

                # Pass 5D actions directly to worker environments
                obs_batch, rewards, terminations, truncations, _ = vector_env.step(actions_batch)
                dones = terminations | truncations

                total_rewards += rewards
                blue_strategy.record_rewards(rewards)

                if np.any(dones):
                    break

            if hasattr(blue_strategy, "finish"):
                blue_strategy.finish()

            completed_episodes += num_envs
            mean_reward = float(np.mean(total_rewards))
            duration_s = round(episode_steps * SIMULATION_STEP_SECONDS, 3)

            if completed_episodes % LOG_FREQ == 0:
                csv_writer.writerow([
                    completed_episodes, round(mean_reward, 4), episode_steps, duration_s, "batch_complete", 1
                ])
                log_file.flush()

            if completed_episodes % 100 == 0:
                elapsed = time.time() - start_time
                sps = (completed_episodes * episode_steps) / elapsed
                print(
                    f"Ep {completed_episodes}/{num_episodes} | Mean Return: {mean_reward:.3f} | "
                    f"Steps: {episode_steps} ({duration_s}s) | Throughput: {sps:.1f} steps/s | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            if completed_episodes % SAVE_FREQ == 0:
                os.makedirs(WEIGHTS_DIR_THIS_RUN, exist_ok=True)
                os.makedirs(WEIGHTS_DIR, exist_ok=True)
                blue_strategy.save_checkpoint(
                    f"{WEIGHTS_DIR_THIS_RUN}/episode_{completed_episodes:06d}.pt", completed_episodes
                )
                blue_strategy.save_checkpoint(f"{WEIGHTS_DIR}/latest.pt", completed_episodes)

    finally:
        log_file.close()
        vector_env.close()
        print(f"Parallel training finished. Logs saved to '{log_filename}'")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallelized RL training runner for SSL Striker tasks."
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="Number of parallel worker processes")
    parser.add_argument("--max-duration-seconds", type=float, default=DEFAULT_EPISODE_DURATION_SECONDS)
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--log-filename", type=str, default=DEFAULT_LOG_FILENAME)
    parser.add_argument("--blue-strategy", default="ssl_strategy_api.strategies.striker:StrikerStrategy")
    parser.add_argument("--yellow-strategy",
                        default="ssl_strategy_api.strategies.idle_goalkeeper_rl:IdleGoalkeeperStrategy")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--task", type=str, choices=["touch", "shoot"], default="shoot")

    args = parser.parse_args()

    blue_strategy = _load_strategy(args.blue_strategy, checkpoint_path=args.weights, num_envs=args.num_envs)
    yellow_strategy = _load_strategy(args.yellow_strategy)

    train_vectorized(
        blue_strategy=blue_strategy,
        yellow_strategy=yellow_strategy,
        num_episodes=args.episodes,
        num_envs=args.num_envs,
        task=args.task,
        max_duration_seconds=args.max_duration_seconds,
        log_filename=args.log_filename,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())