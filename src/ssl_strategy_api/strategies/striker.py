import math
from typing import Mapping
from collections import Counter
import numpy as np

from ssl_strategy_api import RobotState
from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState

######################
# helping variables
######################

# for the field
goal_x = 4.5


class StrikerStrategy:
    """
    a rule-based strategy to score goals by altering different phases:
    - APPROACH
        - drive and turn towards the ball and end the approach phase if the ball is sensed by infrared
    - CARRY
        - take the ball to the position (3.0, 0.0) that is right in front of the enemy's goal.
        - if the ball is lost during the carrying, the phase is reseted to APPROACH
        - when the position is almost reached, check where the goalie is positioned and decide for the opposite side as the aiming target 
        - start the TURNING phase when the position is almost reached
    - TURNING
        - take the decided aim and face this direction
        - if the robo hasn't moved and turned for a short period change to the SHOOTING phase
    - SHOOTING
        - shoot with 5m/s and reset the phase to APPROACH
    """

    def __init__(self, debug = True):
        self.debug = debug
        self.phase_steps = Counter()
        self.active_goalie = True
        self.reset()

    def reset(self, seed=None) -> None:
        self.phase = "APPROACH"
        self.log(self.phase)
        self.aim_y = 0.0                # default aim y-coordinate is the goal's center
        self.steady_steps = 0

    def log(self, text):
        if self.debug:
            print(text)

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        """
        returns the intent for our striker based on the phase it's currently in
        """
        
        if not config.controlled_robot_ids:
            return {}

        if config.team != TeamColor.BLUE:
            return {}

        self.phase_steps[self.phase] += 1 

        robot_id = config.controlled_robot_ids[0]
        robots = state.robots_blue
        robot = robots.get(robot_id)

        if robot is None:
            return {}

        goalie = state.robots_yellow[0]
        if goalie.y_m < -2.5 and sum(self.phase_steps.values()) > 40: # check if the goalie is still in the corner after 40 steps
            self.active_goalie = False
        
##################### APPROACH #############################################
        if self.phase == "APPROACH":
            dx = state.ball.x_m - robot.x_m
            dy = state.ball.y_m - robot.y_m
            distance = (dx * dx + dy * dy) ** 0.5                                   # calc the euclidian distance to the ball position
            target_angle = math.atan2(dy, dx)                                       # calc the direction vector to the ball
            angle_error = target_angle - robot.heading_rad                          # calc the difference in the direction vector and robo's heading
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # choosing the smallest error
            speed_approach = min(2.0, max(0.3, 4.0 * distance))

            intent = RobotIntent(
            vx_mps = speed_approach * dx / distance,
            vy_mps = speed_approach * dy / distance,
            angular_velocity_rad_s = max(-3, min(3, 4 * angle_error)),
            dribbler = distance < 0.3)

            if robot.infrared:
                if state.ball.x_m < 4 and -2.5 < state.ball.y_m < 2.5 and self.active_goalie == False:
                    self.phase = "TURNING"
                    self.log(self.phase)
                else:
                    self.phase = "CARRY"
                    self.log(self.phase)

##################### CARRY ################################################
        elif self.phase == "CARRY":
            dx_reposition = 3 - robot.x_m
            dy_reposition = 0 - robot.y_m
            dx_goal = goal_x - robot.x_m
            dy_goal = 0 - robot.y_m
            distance_reposition = (dx_reposition * dx_reposition + dy_reposition * dy_reposition) ** 0.5 # calc the euclidian distance to the shooting position
            speed_reposition = min(2.0, max(0.3, 4.0 * distance_reposition))
            target_angle_goal = math.atan2(dy_goal, dx_goal)
            angle_to_goal = target_angle_goal - robot.heading_rad
            angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal)) # choosing the smallest error

            intent = RobotIntent(
                vx_mps = speed_reposition * dx_reposition / distance_reposition,
                vy_mps = speed_reposition * dy_reposition / distance_reposition,
                angular_velocity_rad_s = max(-3, min(3, 8 * angle_to_goal)),
                dribbler = True
                )
            
            """ 
            # if the robo is carrying a long way and has a good position to shoot, it shall shoot
            if robot.x_m < 4: 
                h = robot.heading_rad
                w = robot.angular_velocity_rad_s or 0.0
                bvx = 5 * math.cos(h) + robot.vx_mps   # ball speed after shot
                bvy = 5 * math.sin(h) + robot.vy_mps
                if bvx > 0 and abs(w) < 0.3:
                    t = (goal_x - state.ball.x_m) / bvx     # time to goal
                    cross_y = state.ball.y_m + bvy * t      # where it would hit the goal on y-axis
                    if abs(cross_y) < 0.35:
                        self.log("early shot tried")
                        self.phase_steps["EARLY SHOT"] += 1
                        intent = RobotIntent(
                            kick_speed_mps=5,
                            dribbler=False)
                        self.reset()
                        return {robot_id: intent}
            """

            if distance_reposition < 0.4:
                self.phase = "TURNING"
                if abs(goalie.y_m) > 1.0:
                    self.aim_y = 0.0
                elif goalie.y_m < 0:
                    self.aim_y = 0.35
                else:
                    self.aim_y = -0.35
            if not robot.infrared: 
                self.phase = "APPROACH"
                self.log(self.phase)

##################### TURNING ##############################################
        elif self.phase == "TURNING":
            dx_goal = 4.5 - robot.x_m
            dy_goal = self.aim_y - robot.y_m
            target_angle_goal = math.atan2(dy_goal, dx_goal)
            angle_to_goal = target_angle_goal - robot.heading_rad
            angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal)) # choosing the smallest error
            speed = math.hypot(robot.vx_mps or 0.0, robot.vy_mps or 0.0)
            turning = abs(robot.angular_velocity_rad_s or 0.0)
            aligned = abs(angle_to_goal) < 0.075
            calm = speed < 0.075 and turning < 0.15

            intent = RobotIntent(
                angular_velocity_rad_s = max(-3, min(3, 8 * angle_to_goal)),
                dribbler=True
                )

            if aligned and calm:
                self.steady_steps += 1
            else:
                self.steady_steps = 0

            if self.steady_steps >= 2:      # 2 × 25 ms = 0,05s steady
                self.phase = "SHOOTING"
                self.log(self.phase)
                
            if not robot.infrared:
                self.phase = "APPROACH"
                self.log(self.phase)

##################### SHOOTING #############################################
        elif self.phase == "SHOOTING":
            intent = RobotIntent(
            vx_mps=0,
            vy_mps=0,
            angular_velocity_rad_s = 0.0,
            kick_speed_mps=5,
            dribbler=False)
            self.reset()

        return {robot_id: intent}
