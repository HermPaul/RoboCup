from math import atan2, hypot, pi
from typing import Mapping

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState


class GoalkeeperStrategy:
    """Move along the goal line to block the ball's predicted crossing."""

    def __init__(
        self,
        goal_line_offset_m: float = 0.14,
        max_speed_mps: float = 1.8,
        angular_gain: float = 3.0,
        max_angular_speed_rad_s: float = 3.0,
        goal_post_margin_m: float = 0.1,
    ) -> None:
        self.goal_line_offset_m = goal_line_offset_m
        self.max_speed_mps = max_speed_mps
        self.angular_gain = angular_gain
        self.max_angular_speed_rad_s = max_angular_speed_rad_s
        self.goal_post_margin_m = goal_post_margin_m

    def reset(self, seed=None) -> None:
        pass

    def decide(
        self, state: WorldState, config: TeamConfig
    ) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids:
            return {}

        robot_id = config.controlled_robot_ids[0]
        robots = (
            state.robots_blue
            if config.team == TeamColor.BLUE
            else state.robots_yellow
        )
        robot = robots.get(robot_id)
        if robot is None:
            return {robot_id: RobotIntent()}

        defense_direction = -config.attack_direction_x
        goal_line_x = defense_direction * state.field.length_m / 2
        target_x = goal_line_x - defense_direction * self.goal_line_offset_m
        max_goalie_y = max(
            0.0, state.field.goal_width_m / 2 - self.goal_post_margin_m
        )
        target_y = 0.0

        ball = state.ball
        if ball is not None:
            target_y = _clamp(ball.y_m, -max_goalie_y, max_goalie_y)
            if ball.vx_mps is not None and ball.vy_mps is not None:
                is_heading_to_defended_goal = (
                    ball.vx_mps * defense_direction > 0
                )
                if is_heading_to_defended_goal:
                    time_to_goalie = (target_x - ball.x_m) / ball.vx_mps
                    if time_to_goalie > 0:
                        predicted_y = ball.y_m + ball.vy_mps * time_to_goalie
                        target_y = _clamp(
                            predicted_y, -max_goalie_y, max_goalie_y
                        )

        dx = target_x - robot.x_m
        dy = target_y - robot.y_m
        distance = hypot(dx, dy)
        if distance <= 0.025:
            vx_mps = 0.0
            vy_mps = 0.0
        else:
            speed = min(self.max_speed_mps, 2.0 * distance)
            vx_mps = speed * dx / distance
            vy_mps = speed * dy / distance

        desired_heading = 0.0 if config.attack_direction_x > 0 else pi
        angular_velocity = 0.0
        if robot.heading_rad is not None:
            heading_error = _wrap_angle(desired_heading - robot.heading_rad)
            angular_velocity = _clamp(
                self.angular_gain * heading_error,
                -self.max_angular_speed_rad_s,
                self.max_angular_speed_rad_s,
            )

        return {
            robot_id: RobotIntent(
                vx_mps=vx_mps,
                vy_mps=vy_mps,
                angular_velocity_rad_s=angular_velocity,
            )
        }


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi