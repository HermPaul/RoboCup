import argparse
from importlib import import_module
from math import ceil, isfinite
import time
from typing import Any, Optional

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

from ssl_strategy_api import FieldGeometry, TeamColor, TeamConfig
from ssl_strategy_api.adapters.rsoccer import (
    to_rsoccer_commands,
    world_state_from_rsoccer_frame,
)
from ssl_strategy_api.helpers import BallSpawner



SIMULATION_STEP_SECONDS = 0.025
DEFAULT_DURATION_SECONDS = 90.0
BLUE_ROBOT_ID = 0
YELLOW_ROBOT_ID = 0
RIGHT_SIDE_MARGIN_M = 0.20
BALL_IDLE_SPEED_MPS = 0.06


class Exercise1Env(SSLBaseEnv):
    """Exercise environment with one blue striker and one yellow goalkeeper."""

    def __init__(self, spawner_seed: Optional[int] = None, render_mode=None):
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
            render_mode=render_mode,
        )
        self.canonical_field = FieldGeometry(
            length_m=self.field.length,
            width_m=self.field.width,
            goal_width_m=self.field.goal_width,
        )
        self.state = None
        self.ball_spawner = BallSpawner(seed=spawner_seed)

    def _get_commands(self, commands):
        return commands

    def _frame_to_observations(self):
        self.state = world_state_from_rsoccer_frame(
            self.frame,
            self.canonical_field,
            frame_number=self.steps,
        )
        return self.state

    def _calculate_reward_and_done(self):
        ball = self.state.ball if self.state is not None else None
        blue_robot = self.frame.robots_blue.get(BLUE_ROBOT_ID)
        respawn_reason = self.ball_spawner.respawn_when_goal_or_out_or_idle(
            ball,
            self.field,
            attack_direction_x=self.blue_config.attack_direction_x,
            idle_speed_mps=BALL_IDLE_SPEED_MPS,
        )

        blue_in_opponent_goal_area = False
        if blue_robot is not None:
            blue_in_opponent_goal_area = self.ball_spawner.in_opponent_goal_area(
                self.field,
                blue_robot.x,
                blue_robot.y,
                attack_direction_x=self.blue_config.attack_direction_x,
            )

        done = blue_in_opponent_goal_area or respawn_reason is not None

        # Create here your reward function
        reward = 0.0
        # approach reward
        dx = blue_robot.x - ball.x_m
        dy = blue_robot.y - ball.y_m
        dist_robot_ball = (dx ** 2 + dy ** 2) ** 0.5
        reward = -dist_robot_ball # * 0.1 to weigh it smaller

        # check if face of robot is kicking the ball
        if blue_robot.infrared:
            reward = 100
            done = True
        #print(reward)

        # alignment reward 
        # center is (0,0)
        goal_x = (self.canonical_field.length_m / 2.0)
        goal_y = 0.0
        # positional diff ball and goal
        bg_x = goal_x -ball.x_m 
        bg_y = goal_y - ball.y_m 
        dist_bg = (bg_x**2 + bg_y**2) ** 0.5

        rb_x = ball.x_m - blue_robot.x
        rb_y = ball.y_m - blue_robot.y
        dist_rb = (rb_x**2 + rb_y**2) ** 0.5

        angular_alignment = 0.0
        # check if the vectors show in the same direction
        dot_product = (rb_x * bg_x) + (rb_y * bg_y)
        angular_alignment = dot_product / (dist_bg * dist_rb)
        print(angular_alignment)

        reward
    

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


def _load_strategy(path: str) -> Any:
    module_name, separator, class_name = path.partition(":")
    if not separator:
        raise ValueError("strategy must use module.path:ClassName format")
    strategy_class = getattr(import_module(module_name), class_name)
    return strategy_class()


def run_demo(
    blue_strategy: Any,
    yellow_strategy: Any,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    render: bool = True,
    seed: Optional[int] = None,
) -> None:
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")

    env = Exercise1Env(spawner_seed=seed, render_mode="human" if render else None)
    observation, _ = env.reset(seed=seed)

    blue_reset = getattr(blue_strategy, "reset", None)
    yellow_reset = getattr(yellow_strategy, "reset", None)
    if blue_reset is not None:
        blue_reset(seed=seed)
    if yellow_reset is not None:
        yellow_reset(seed=seed)

    episode_started = time.monotonic()
    max_steps = ceil(duration_seconds / SIMULATION_STEP_SECONDS)

    try:
        for step in range(1, max_steps + 1):
            if time.monotonic() >= episode_started + duration_seconds:
                break

            blue_intents = blue_strategy.decide(observation, env.blue_config)
            yellow_intents = yellow_strategy.decide(observation, env.yellow_config)

            commands = to_rsoccer_commands(observation, env.blue_config, blue_intents)
            commands.extend(
                to_rsoccer_commands(observation, env.yellow_config, yellow_intents)
            )

            observation, reward, done, _, _ = env.step(commands)
            if done:
                elapsed = time.monotonic() - episode_started
                print(
                    "Episode ended at step {} ({:.2f}s), reward={:.3f}".format(
                        step,
                        elapsed,
                        reward,
                    ),
                    flush=True,
                )
                break

            next_tick = episode_started + step * SIMULATION_STEP_SECONDS
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        else:
            elapsed = time.monotonic() - episode_started
            print(
                "Episode reached time limit after {} steps ({:.2f}s).".format(
                    max_steps,
                    elapsed,
                ),
                flush=True,
            )
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run rSoccer exercise 1 with configurable blue/yellow strategies."
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the rSoccer field (default: on; use --no-render for headless runs)",
    )

    # You can change these defaults to bind any other strategy classes.
    parser.add_argument(
        "--blue-strategy",
        default="ssl_strategy_api.strategies.striker:StrikerStrategy",
        help="Blue strategy class as module.path:ClassName",
    )
    parser.add_argument(
        "--yellow-strategy",
        default="ssl_strategy_api.strategies.idle_goalkeeper_rl:IdleGoalkeeperStrategy",
        help="Yellow strategy class as module.path:ClassName",
    )

    # You could parse instead the classic goalkeeper by using the following default:
    # parser.add_argument(
    #     "--yellow-strategy",
    #     default="ssl_strategy_api.strategies.goalkeeper:GoalkeeperStrategy",
    #     help="Yellow strategy class as module.path:ClassName",
    # )

    # Or you could parse a RL-based goalkeeper by using the following default:
    # parser.add_argument(
    #     "--yellow-strategy",
    #     default="ssl_strategy_api.strategies.goalkeeper_rl:QTableGoalkeeperStrategy",
    #     help="Yellow strategy class as module.path:ClassName",
    # )
    args = parser.parse_args()

    if not isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")

    # Default binding for this exercise:
    #   Blue team  -> striker (student side)
    #   Yellow team -> goalkeeper (opponent side)
    # Swap the strategy class strings above or pass CLI args to change behavior.
    blue_strategy: Any = _load_strategy(args.blue_strategy)
    yellow_strategy: Any = _load_strategy(args.yellow_strategy)

    run_demo(
        blue_strategy=blue_strategy,
        yellow_strategy=yellow_strategy,
        duration_seconds=args.duration_seconds,
        render=args.render,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
