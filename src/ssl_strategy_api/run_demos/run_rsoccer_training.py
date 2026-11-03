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
DEFAULT_EPISODE_DURATION_SECONDS = 8.0
BLUE_ROBOT_ID = 0
YELLOW_ROBOT_ID = 0
RIGHT_SIDE_MARGIN_M = 0.20
DEFAULT_NUM_EPISODES = 500000
SAVE_FREQ = 10000
NUM_ENVS = 128
ROLLOUT_STEPS = 20
WEIGHTS_DIR = "ssl_strategy_api/src/ssl_strategy_api/strategies/weights/striker/"
WEIGHTS_DIR_THIS_RUN = f"{WEIGHTS_DIR}run_{time.strftime('%Y%m%d_%H%M%S')}/"
DEFAULT_LOG_FILENAME = f"ssl_strategy_api/logs/run_{time.strftime('%Y%m%d_%H%M%S')}.csv"
LOG_FREQ = 500


class TrainingEnv(SSLBaseEnv):

    def __init__(
        self,
        spawner_seed: Optional[int] = None,
        evaluator: Optional[Any] = None,
        max_duration_seconds: float = DEFAULT_EPISODE_DURATION_SECONDS,
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

        self.max_steps = ceil(max_duration_seconds / SIMULATION_STEP_SECONDS)
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

        raise ValueError(f"Invalid action passed: {type(action)}")

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

        dx_ball = ball.x_m - striker.x_m
        dy_ball = ball.y_m - striker.y_m
        dist_ball = float(np.hypot(dx_ball, dy_ball))

        angle_ball = float(np.arctan2(dy_ball, dx_ball) - striker.heading_rad)
        angle_ball = float(np.arctan2(np.sin(angle_ball), np.cos(angle_ball)))

        goal_x = (self.field.length / 2.0) * self.blue_config.attack_direction_x
        dx_goal = goal_x - striker.x_m
        dy_goal = 0.0 - striker.y_m

        dist_goal = float(np.hypot(dx_goal, dy_goal))

        dx_gk = goalkeeper.x_m - striker.x_m
        dy_gk = goalkeeper.y_m - striker.y_m

        return np.array([
            dx_ball, dy_ball, dist_ball, angle_ball,
            striker.heading_rad, striker.vx_mps, striker.vy_mps, striker.angular_velocity_rad_s,
            float(striker.infrared or 0),
            dx_goal, dy_goal, dist_goal,
            dx_gk, dy_gk, float(np.hypot(goalkeeper.vx_mps, goalkeeper.vy_mps))
        ], dtype=np.float32)

    def _calculate_reward_and_done(self):
        reward, done, info = self.evaluator.calculate_reward_and_done(self)
        if self.steps >= self.max_steps:
            done = True
            info["finish_reason"] = "max_duration_reached"

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
            x=ball_spawn.x_m, y=ball_spawn.y_m, v_x=ball_spawn.vx_mps, v_y=ball_spawn.vy_mps
        )
        frame.robots_blue[BLUE_ROBOT_ID] = Robot(
            yellow=False, id=BLUE_ROBOT_ID, x=blue_spawn.x_m, y=blue_spawn.y_m, theta=180.0
        )

        half_length = self.field.length / 2
        half_width = self.field.width / 2
        frame.robots_yellow[YELLOW_ROBOT_ID] = Robot(
            yellow=True, id=YELLOW_ROBOT_ID, x=half_length - 0.10, y=-(half_width), theta=180.0
        )
        return frame


def make_env_factory(
    seed: Optional[int],
    env_id: int,
    task: str,
    max_duration_seconds: float = DEFAULT_EPISODE_DURATION_SECONDS,
):
    def _thunk():
        evaluator = TouchAndShootEvaluator() if task == "shoot" else TouchEvaluator()
        env_seed = (seed + env_id) if seed is not None else None
        return TrainingEnv(
            spawner_seed=env_seed,
            evaluator=evaluator,
            max_duration_seconds=max_duration_seconds,
        )
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
    rollout_steps: int = 128,  # Increased from 20 to 128 to reduce GPU-CPU sync overhead
    task: str = "shoot",
    max_duration_seconds: float = DEFAULT_EPISODE_DURATION_SECONDS,
    log_filename: str = DEFAULT_LOG_FILENAME,
    seed: Optional[int] = None,
) -> None:
    env_fns = [
        make_env_factory(seed, i, task, max_duration_seconds) for i in range(num_envs)
    ]
    vector_env = gym.vector.AsyncVectorEnv(env_fns)

    os.makedirs(os.path.dirname(log_filename) or ".", exist_ok=True)
    log_file = open(log_filename, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "episode", "total_reward", "steps", "duration_seconds", "finish_reason", "touched_ball"
    ])

    print(f"Parallel training active on {num_envs} envs. Target: {num_episodes} episodes...")
    start_time = time.time()
    completed_episodes = 0
    total_env_steps = 0
    last_saved_ep = 0
    last_logged_ep = 0

    try:
        obs_batch, _ = vector_env.reset(seed=seed)

        while completed_episodes < num_episodes:
            total_rewards = np.zeros(num_envs)

            for _ in range(rollout_steps):
                actions_batch = blue_strategy.decide_batch(obs_batch)

                obs_batch, rewards, terminations, truncations, _ = vector_env.step(actions_batch)
                dones = terminations | truncations

                blue_strategy.record_step(rewards, dones)
                total_rewards += rewards
                completed_episodes += int(np.sum(dones))

            total_env_steps += rollout_steps * num_envs

            # Execute gradient update pass
            blue_strategy.finish_rollout(obs_batch)

            mean_reward = float(np.mean(total_rewards))

            # FIXED: Write logs ONLY when hitting the logging threshold
            if completed_episodes - last_logged_ep >= LOG_FREQ:
                last_logged_ep = completed_episodes
                elapsed = time.time() - start_time
                true_sps = total_env_steps / max(elapsed, 1.0)

                csv_writer.writerow([
                    completed_episodes, round(mean_reward, 4), rollout_steps, round(elapsed, 1), "rollout_pass", 1
                ])
                log_file.flush()  # Disk write happens every 500 episodes instead of 230x/sec

                print(
                    f"Ep {completed_episodes}/{num_episodes} | Mean Reward: {mean_reward:.2f} | "
                    f"Throughput: {true_sps:.1f} steps/s | Elapsed: {elapsed:.1f}s"
                )

            # Checkpoint saving on episode threshold
            if completed_episodes - last_saved_ep >= SAVE_FREQ:
                last_saved_ep = completed_episodes
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
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--max-duration-seconds", type=float, default=DEFAULT_EPISODE_DURATION_SECONDS)
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--log-filename", type=str, default=DEFAULT_LOG_FILENAME)
    parser.add_argument("--blue-strategy", default="ssl_strategy_api.strategies.striker:StrikerStrategy")
    parser.add_argument("--yellow-strategy", default="ssl_strategy_api.strategies.idle_goalkeeper_rl:IdleGoalkeeperStrategy")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--task", type=str, choices=["touch", "shoot"], default="shoot")

    args = parser.parse_args()

    blue_strategy = _load_strategy(
        args.blue_strategy, checkpoint_path=args.weights, num_envs=args.num_envs
    )
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