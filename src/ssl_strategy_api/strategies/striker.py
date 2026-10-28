import math
from typing import Mapping

import numpy as np

from ssl_strategy_api import RobotState
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

        dx = state.ball.x_m - robot.x_m
        dy = state.ball.y_m - robot.y_m
        angle = math.atan2(dy, dx)

        return {
            robot_id: RobotIntent(
                #vx_mps=0,
                #vy_mps=1,
                #angular_velocity_rad_s=3,

                vx_mps=dx/(abs(dx+dy)) * 2,
                vy_mps=dy/(abs(dx+dy)) * 2,
                angular_velocity_rad_s=max(-3, min(3, 5*math.atan2(angle - robot.heading_rad, math.cos(angle - robot.heading_rad)))),
                kick_speed_mps=5 if robot.infrared else 0,
            )
        }

    #def observe(self, state: WorldState, config: TeamConfig, striker: RobotState) -> np.ndarray:



