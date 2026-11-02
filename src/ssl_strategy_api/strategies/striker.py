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

    def __init__(self, debug = True):
        self.debug = debug
        self.reset()

    def reset(self, seed=None) -> None:
        self.phase = "APPROACH"
        self.log(self.phase)
        self.aim_y = 0.0 # default aim y-coordinate is the goal's center
        self.steady_steps = 0

    def log(self, text):
        if self.debug:
            print(text)

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
        goalie = state.robots_yellow[0]

        if self.phase == "APPROACH":
            dx = state.ball.x_m - robot.x_m
            dy = state.ball.y_m - robot.y_m
            distance = (dx * dx + dy * dy) ** 0.5
            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - robot.heading_rad
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error)) # choosing the smallest error
            speed_approach = min(2.0, max(0.3, 4.0 * distance))

            intent = RobotIntent(
            vx_mps = speed_approach * dx / distance,
            vy_mps = speed_approach * dy / distance,
            angular_velocity_rad_s = max(-3, min(3, 4 * angle_error)),
            dribbler = distance < 0.3)

            if robot.infrared:
                self.log(self.phase)
                self.phase = "CARRY"


        elif self.phase == "CARRY":
            dx_repos = 3 - robot.x_m
            dy_repos = 0 - robot.y_m
            dx_goal = 4.5 - robot.x_m
            dy_goal = 0 - robot.y_m
            distance_repos = (dx_repos * dx_repos + dy_repos * dy_repos) ** 0.5
            speed_repos = min(2.0, max(0.3, 4.0 * distance_repos))
            target_angle_goal = math.atan2(dy_goal, dx_goal)
            angle_to_goal = target_angle_goal - robot.heading_rad
            angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal)) # choosing the smallest error

            intent = RobotIntent(
            vx_mps = speed_repos * dx_repos / distance_repos,
            vy_mps = speed_repos * dy_repos / distance_repos,
            angular_velocity_rad_s = max(-3, min(3, 4 * angle_to_goal)),
            dribbler = True)

            if distance_repos < 0.2:
                self.phase = "TURNING"
                y_goalie = goalie.y_m
                if abs(y_goalie) > 1.0:
                    self.aim_y = 0.0
                    self.log("shooting direction: center")
                elif y_goalie < 0:
                    self.aim_y = 0.35
                    #self.log("shooting direction: left")
                else:
                    self.aim_y = -0.35
                    #self.log("shooting direction: right")
            if not robot.infrared: 
                self.phase = "APPROACH"
                self.log(self.phase)


        elif self.phase == "TURNING":
            dx_goal = 4.5 - robot.x_m
            dy_goal = self.aim_y - robot.y_m
            target_angle_goal = math.atan2(dy_goal, dx_goal)
            angle_to_goal = target_angle_goal - robot.heading_rad
            angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal)) # choosing the smallest error

            speed = math.hypot(robot.vx_mps or 0.0, robot.vy_mps or 0.0)
            turning = abs(robot.angular_velocity_rad_s or 0.0)

            aligned = abs(angle_to_goal) < 0.05
            calm = speed < 0.05 and turning < 0.1

            intent = RobotIntent(
            vx_mps=0,
            vy_mps=0,
            angular_velocity_rad_s = max(-3, min(3, 4 * angle_to_goal)),
            dribbler=True)

            if aligned and calm:
                self.steady_steps += 1
            else:
                self.steady_steps = 0

            if self.steady_steps >= 5:      # 5 × 25 ms = 0,125s steady
                self.phase = "SHOOTING"
                self.log(self.phase)
                
            if not robot.infrared:
                self.phase = "APPROACH"
                self.log(self.phase)
        
        elif self.phase == "SHOOTING":
            intent = RobotIntent(
            vx_mps=0,
            vy_mps=0,
            angular_velocity_rad_s = 0.0,
            kick_speed_mps=5,
            dribbler=False)

            self.reset()

        return {robot_id: intent}

    #def observe(self, state: WorldState, config: TeamConfig, striker: RobotState) -> np.ndarray:
