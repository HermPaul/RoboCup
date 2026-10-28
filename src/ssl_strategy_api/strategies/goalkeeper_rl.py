"""Goalkeeper strategy that plays with the Q-table trained by goalkeeper_rl_example.py.

QTableGoalkeeperStrategy follows the API's ResettableStrategy protocol from
ssl_strategy_api/strategy.py (decide() and reset()), so any API runner can use
it in place of the rule-based GoalkeeperStrategy. It needs only the API and numpy.
"""

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from ssl_strategy_api import RobotIntent, TeamColor, TeamConfig, WorldState

# These must match the values used in training (goalkeeper_rl_example.py).
TABLE_PATH = Path(__file__).parent / "goalkeeper_q_table.npy"  # Next to this file.
GOALKEEPER_LINE_OFFSET_M = 0.14
MOVE_SPEEDS = [0.0, 0.8, -0.8]  # stay, left (+y), right (-y), in m/s.
BIN_EDGES = [-0.4, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.4]


class QTableGoalkeeperStrategy:
    """Picks the goalkeeper's move from a trained Q-table."""

    def __init__(self, table_path: str = TABLE_PATH):
        self.q_table = np.load(table_path)

    def reset(self, seed: Optional[int] = None) -> None:
        """Nothing to reset: the table does not change while playing."""

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        """Returns the move with the highest table value for the current state.

        Args:
            state: The current world, in meters and radians.
            config: The goalkeeper's team, robot ID and attack direction.

        Returns:
            {robot_id: RobotIntent} for the goalkeeper.
        """
        robot_id = config.controlled_robot_ids[0]
        robots = state.robots_blue if config.team == TeamColor.BLUE else state.robots_yellow
        goalkeeper = robots.get(robot_id)
        if goalkeeper is None or state.ball is None:
            return {robot_id: RobotIntent()}  # Nothing to react to: stand still.

        _, spot_y = best_spot(state, config)
        row = int(np.digitize(spot_y - goalkeeper.y_m, BIN_EDGES))
        move = int(np.argmax(self.q_table[row]))
        return {robot_id: RobotIntent(vy_mps=MOVE_SPEEDS[move])}


def best_spot(state: WorldState, config: TeamConfig):
    """Finds where the goalkeeper should stand (same formula as in training).

    Args:
        state: The current world.
        config: The goalkeeper's team configuration.

    Returns:
        The spot as (x, y) in meters.
    """
    ball = state.ball
    field = state.field
    defense_direction = -config.attack_direction_x  # -1 means our goal is at negative x.
    goal_line_x = defense_direction * field.length_m / 2
    spot_x = goal_line_x - defense_direction * GOALKEEPER_LINE_OFFSET_M
    spot_y = ball.y_m * field.goal_width_m / field.width_m

    moving_toward_our_side = ball.vx_mps is not None and ball.vx_mps * defense_direction > 0
    if moving_toward_our_side:
        time_to_goal = (goal_line_x - ball.x_m) / ball.vx_mps
        y_at_goal = ball.y_m + ball.vy_mps * time_to_goal
        if time_to_goal > 0 and abs(y_at_goal) <= field.goal_width_m / 2:
            time_to_goalkeeper = (spot_x - ball.x_m) / ball.vx_mps
            spot_y = ball.y_m + ball.vy_mps * time_to_goalkeeper
    return spot_x, spot_y

