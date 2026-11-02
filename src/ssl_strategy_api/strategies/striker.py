import math
from typing import Mapping

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState


class StrikerStrategy:
    """Student placeholder for Exercise 1.

    Replace this strategy with your own implementation.

    The default behavior below is intentionally simple and deliberately not optimal:
    it demonstrates how to access the controlled robot and return a valid mapping.
    """

    def reset(self, seed=None) -> None:
        pass

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        """Simple starter example for students.

        You must change this strategy to implement your own striker behavior.

        This example shows the basic structure:
        - check whether any robots are controlled
        - select the robot for the blue team
        - get its state from the world
        - return a mapping for that robot id

        In this exercise, the blue team is the striker and uses robot id 0 by default.
        """
        if not config.controlled_robot_ids:
            return {}

        if config.team != TeamColor.BLUE:
            return {}

        robot_id = config.controlled_robot_ids[0]
        robots = state.robots_blue
        robot = robots.get(robot_id)
        if robot is None:
            return {}

        # Simple motion example: move the robot in a small random-like direction.
        # Replace this with your own strategy logic.
        theta = state.frame_number * 0.1 if state.frame_number is not None else 0.0
        return {
            robot_id: RobotIntent(
                vx_mps=0.8 * math.cos(theta),
                vy_mps=0.8 * math.sin(theta),
                angular_velocity_rad_s=1.0,
            )
        }
