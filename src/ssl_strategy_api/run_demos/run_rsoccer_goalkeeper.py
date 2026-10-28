import argparse
from math import ceil, isfinite
import time
from typing import Optional

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

from ssl_strategy_api import FieldGeometry, TeamColor, TeamConfig
from ssl_strategy_api.adapters.rsoccer import (
    to_rsoccer_commands,
    world_state_from_rsoccer_frame,
)
from ssl_strategy_api.strategies.goalkeeper import GoalkeeperStrategy
from ssl_strategy_api.helpers import BallSpawner, GoalTracker


SIMULATION_STEP_SECONDS = 0.025
DEFAULT_DURATION_SECONDS = 90.0
GOALKEEPER_TEAM = TeamColor.BLUE
GOALKEEPER_ID = 0
ATTACK_DIRECTION_X = 1


class GoalkeeperDemoEnv(SSLBaseEnv):
    """One-goalkeeper SSL drill with a randomized, incoming ball."""

    def __init__(self, ball_spawner: BallSpawner, render_mode=None):
        self.team_config = TeamConfig(
            team=GOALKEEPER_TEAM,
            controlled_robot_ids=(GOALKEEPER_ID,),
            attack_direction_x=ATTACK_DIRECTION_X,
        )
        self.ball_spawner = ball_spawner
        super().__init__(
            field_type=1,
            n_robots_blue=1 if GOALKEEPER_TEAM == TeamColor.BLUE else 0,
            n_robots_yellow=1 if GOALKEEPER_TEAM == TeamColor.YELLOW else 0,
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
        return world_state_from_rsoccer_frame(
            self.frame,
            self.canonical_field,
            frame_number=self.steps,
        )

    def _calculate_reward_and_done(self):
        return 0.0, False

    def _get_initial_positions_frame(self):
        frame = Frame()
        spawn = self.ball_spawner.spawn_and_shoot_towards_goal(
            self.canonical_field,
            self.team_config,
        )
        frame.ball = Ball(
            x=spawn.x_m,
            y=spawn.y_m,
            v_x=spawn.vx_mps,
            v_y=spawn.vy_mps,
        )

        defense_direction = -self.team_config.attack_direction_x
        goalie_x = defense_direction * (self.field.length / 2 - 0.14)
        goalie_theta = 0.0 if self.team_config.attack_direction_x > 0 else 180.0
        robot = Robot(
            yellow=self.team_config.team == TeamColor.YELLOW,
            id=GOALKEEPER_ID,
            x=goalie_x,
            y=0.0,
            theta=goalie_theta,
        )
        if self.team_config.team == TeamColor.BLUE:
            frame.robots_blue[GOALKEEPER_ID] = robot
        else:
            frame.robots_yellow[GOALKEEPER_ID] = robot
        return frame


def run_demo(
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    stop_after_goals: Optional[int] = None,
    render: bool = True,
    seed: Optional[int] = None,
) -> int:
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if stop_after_goals is not None and stop_after_goals < 1:
        raise ValueError("stop_after_goals must be positive when provided")

    spawner = BallSpawner(seed=seed)
    env = GoalkeeperDemoEnv(
        spawner,
        render_mode="human" if render else None,
    )
    strategy = GoalkeeperStrategy()
    score = GoalTracker()
    observation, _ = env.reset(seed=seed)
    score.reset_ball_tracking(observation.ball)
    strategy.reset(seed=seed)
    episode_started = time.monotonic()
    max_steps = ceil(duration_seconds / SIMULATION_STEP_SECONDS)
    steps_run = 0
    respawns = 0

    try:
        for step in range(1, max_steps + 1):
            if time.monotonic() >= episode_started + duration_seconds:
                break
            steps_run = step
            intents = strategy.decide(observation, env.team_config)
            commands = to_rsoccer_commands(observation, env.team_config, intents)
            observation, _, _, _, _ = env.step(commands)

            scorer = score.update(observation.ball, observation.field, env.team_config)
            if scorer is not None and stop_after_goals is not None:
                if score.total_goals >= stop_after_goals:
                    break

            reason = spawner.respawn_when_idle_or_out(
                observation.ball,
                observation.field,
            )
            if scorer is not None or reason is not None:
                if reason is not None:
                    print("Ball {}; respawning a new shot.".format(reason), flush=True)
                observation, _ = env.reset()
                score.reset_ball_tracking(observation.ball)
                strategy.reset(seed=None)
                respawns += 1

            next_tick = episode_started + step * SIMULATION_STEP_SECONDS
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        elapsed = time.monotonic() - episode_started
        print(
            "Goalkeeper episode finished: {:.1f} seconds, {} steps, {} respawn(s); {}.".format(
                elapsed, steps_run, respawns, score.score_line
            ),
            flush=True,
        )
        return score.total_goals
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the rule-based goalkeeper drill in rSoccer SSL."
    )
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--stop-after-goals", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the rSoccer field (default: on; use --no-render for headless runs)",
    )
    args = parser.parse_args()
    if not isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.stop_after_goals is not None and args.stop_after_goals < 1:
        parser.error("--stop-after-goals must be positive")

    goals = run_demo(
        duration_seconds=args.duration_seconds,
        stop_after_goals=args.stop_after_goals,
        render=args.render,
        seed=args.seed,
    )
    return 0 if args.stop_after_goals is None or goals >= args.stop_after_goals else 1


if __name__ == "__main__":
    raise SystemExit(main())