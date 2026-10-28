import argparse
from importlib import import_module
from math import ceil, isfinite
import time
from typing import Any, Dict, Optional

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

from ssl_strategy_api import FieldGeometry, TeamColor, TeamConfig
from ssl_strategy_api.adapters.rsoccer import (
    to_rsoccer_commands,
    world_state_from_rsoccer_frame,
)
from ssl_strategy_api.helpers import BallSpawner, GoalTracker


SIMULATION_STEP_SECONDS = 0.025
DEFAULT_DURATION_SECONDS = 90.0
BLUE_ROBOT_ID = 0
YELLOW_ROBOT_ID = 0
RIGHT_SIDE_MARGIN_M = 0.20


class Exercise1EvalEnv(SSLBaseEnv):
    """Exercise1 evaluation env: same setup as exercise1, no stop conditions."""

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
        self.ball_spawner = BallSpawner(seed=spawner_seed)
        self.state = None
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
        # Evaluation runner controls respawns externally and keeps one long episode.
        return 0.0, False

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

    def respawn_ball_keep_robots(self) -> None:
        """Respawn only the ball; keep both robots at current simulator positions."""
        spawn = self.ball_spawner.spawn_stationary_right_side(
            self.field,
            attack_direction_x=self.blue_config.attack_direction_x,
            right_side_margin_m=RIGHT_SIDE_MARGIN_M,
            enforce_outside_opponent_goal_area=True,
        )

        frame = Frame()
        frame.ball = Ball(x=spawn.x_m, y=spawn.y_m, v_x=spawn.vx_mps, v_y=spawn.vy_mps)

        current_blue = self.frame.robots_blue.get(BLUE_ROBOT_ID)
        if current_blue is not None:
            frame.robots_blue[BLUE_ROBOT_ID] = Robot(
                yellow=False,
                id=current_blue.id,
                x=current_blue.x,
                y=current_blue.y,
                theta=current_blue.theta,
            )

        current_yellow = self.frame.robots_yellow.get(YELLOW_ROBOT_ID)
        if current_yellow is not None:
            frame.robots_yellow[YELLOW_ROBOT_ID] = Robot(
                yellow=True,
                id=current_yellow.id,
                x=current_yellow.x,
                y=current_yellow.y,
                theta=current_yellow.theta,
            )

        self.rsim.reset(frame)
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()


def _load_strategy(path: str) -> Any:
    module_name, separator, class_name = path.partition(":")
    if not separator:
        raise ValueError("strategy must use module.path:ClassName format")
    strategy_class = getattr(import_module(module_name), class_name)
    return strategy_class()


def run_evaluation(
    blue_strategy: Any,
    yellow_strategy: Any,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    render: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")

    env = Exercise1EvalEnv(spawner_seed=seed, render_mode="human" if render else None)
    observation, _ = env.reset(seed=seed)

    blue_reset = getattr(blue_strategy, "reset", None)
    yellow_reset = getattr(yellow_strategy, "reset", None)
    if blue_reset is not None:
        blue_reset(seed=seed)
    if yellow_reset is not None:
        yellow_reset(seed=seed)

    stats: Dict[str, int] = {
        "respawns_total": 0,
        "goals": 0,
        "out_of_bounds": 0,
        "idle_in_opponent_goal_area": 0,
        "blue_goal_area_entries": 0,
    }
    score = GoalTracker()
    score.reset_ball_tracking(observation.ball)

    episode_started = time.monotonic()
    max_steps = ceil(duration_seconds / SIMULATION_STEP_SECONDS)
    steps_run = 0

    try:
        for step in range(1, max_steps + 1):
            if time.monotonic() >= episode_started + duration_seconds:
                break
            steps_run = step

            blue_intents = blue_strategy.decide(observation, env.blue_config)
            yellow_intents = yellow_strategy.decide(observation, env.yellow_config)

            commands = to_rsoccer_commands(observation, env.blue_config, blue_intents)
            commands.extend(
                to_rsoccer_commands(observation, env.yellow_config, yellow_intents)
            )
            observation, _, _, _, _ = env.step(commands)

            scorer = score.update(observation.ball, observation.field, env.blue_config)
            if scorer is not None:
                stats["goals"] += 1

            respawn_reason = env.ball_spawner.respawn_when_goal_or_out_or_idle(
                observation.ball,
                env.field,
                attack_direction_x=env.blue_config.attack_direction_x,
            )

            blue_robot = observation.robots_blue.get(BLUE_ROBOT_ID)
            blue_in_opponent_goal_area = False
            if blue_robot is not None:
                blue_in_opponent_goal_area = env.ball_spawner.in_opponent_goal_area(
                    env.field,
                    blue_robot.x_m,
                    blue_robot.y_m,
                    attack_direction_x=env.blue_config.attack_direction_x,
                )
            if blue_in_opponent_goal_area:
                stats["blue_goal_area_entries"] += 1

            if respawn_reason is not None:
                stats["respawns_total"] += 1
                if respawn_reason == "out-of-bounds":
                    stats["out_of_bounds"] += 1
                elif respawn_reason == "idle-in-opponent-goal-area":
                    stats["idle_in_opponent_goal_area"] += 1
                env.respawn_ball_keep_robots()
                observation = env._frame_to_observations()
                score.reset_ball_tracking(observation.ball)

            next_tick = episode_started + step * SIMULATION_STEP_SECONDS
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        elapsed = time.monotonic() - episode_started
        print(
            (
                "Exercise1 evaluation finished: {:.1f}s, {} steps, {} respawns "
                "(goals={}, out_of_bounds={}, idle_in_opponent_goal_area={}, "
                "blue_goal_area_entries={})."
            ).format(
                elapsed,
                steps_run,
                stats["respawns_total"],
                stats["goals"],
                stats["out_of_bounds"],
                stats["idle_in_opponent_goal_area"],
                stats["blue_goal_area_entries"],
            ),
            flush=True,
        )
        return stats
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run exercise1 evaluation for a fixed duration with respawns instead of full stop."
        )
    )
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the rSoccer field (default: on; use --no-render for headless runs)",
    )
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

    blue_strategy = _load_strategy(args.blue_strategy)
    yellow_strategy = _load_strategy(args.yellow_strategy)
    run_evaluation(
        blue_strategy=blue_strategy,
        yellow_strategy=yellow_strategy,
        duration_seconds=args.duration_seconds,
        render=args.render,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
