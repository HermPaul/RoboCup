import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ssl_strategy_api.run_demos.run_rsoccer_training import TrainingEnv

BLUE_ROBOT_ID = 0
BALL_IDLE_SPEED_MPS = 0.06
GOAL_X = 4.5
GOAL_Y = 0
GOAL_SIZE = 1



class TouchAndShootEvaluator:
    """Evaluates per-step dense rewards and finish criteria."""

    def __init__(self):
        self.prev_angle_err: Optional[float] = None
        self.prev_distance: Optional[float] = None
        self.ball_already_touched: bool = False
        self.prev_goal_angular_size: Optional[float] = None

    def calculate_reward_and_done(
            self, env: "TrainingEnv"
    ) -> Tuple[float, bool, Dict[str, Any]]:
        state = env.state
        if state is None:
            return 0.0, False, {"finish_reason": "missing_state", "touched_ball": False}

        striker = state.robots_blue.get(BLUE_ROBOT_ID)
        ball = state.ball

        if striker is None or ball is None:
            return 0.0, False, {"finish_reason": "missing_entities", "touched_ball": False}

        reward = 0.0
        if self.ball_already_touched:
            reward = self.calculate_reward_after_touching(ball, striker)
        else:
            reward = self.calculate_reward_before_touching(ball, striker)
            if striker.infrared:
                self.ball_already_touched = True
                reward += 800.0

        # Environment Terminations
        blue_in_opponent_goal_area = env.ball_spawner.in_opponent_goal_area(
            env.field,
            striker.x_m,
            striker.y_m,
            attack_direction_x=env.blue_config.attack_direction_x,
        )

        respawn_reason = env.ball_spawner.respawn_when_goal_or_out_or_idle(
            ball,
            env.field,
            attack_direction_x=env.blue_config.attack_direction_x,
            idle_speed_mps=BALL_IDLE_SPEED_MPS,
        )

        done = False
        finish_reason = "in_progress"

        if blue_in_opponent_goal_area:
            done = True
            reward -= 50.0
            finish_reason = "goal_area_foul"
        elif respawn_reason is not None:
            done = True
            reward -= 50.0
            finish_reason = f"out_or_idle_{respawn_reason}"

        if done:
            self.prev_distance = None
            self.prev_angle_err = None
            self.ball_already_touched = False
            self.prev_goal_angular_size = None

        info = {
            "finish_reason": finish_reason,
            "touched_ball": self.ball_already_touched,
        }
        return reward, done, info

    def calculate_reward_before_touching(self, ball, striker):
        # 1. Distance Calculation & Progress
        dx = ball.x_m - striker.x_m
        dy = ball.y_m - striker.y_m
        current_dist = float(np.hypot(dx, dy))

        if self.prev_distance is None:
            self.prev_distance = current_dist

        distance_closed = self.prev_distance - current_dist

        # 2. Heading Alignment & Angular Progress
        angle_to_ball = float(np.arctan2(dy, dx) - striker.heading_rad)
        # Normalize angle to [-pi, pi]
        angle_to_ball = float(np.arctan2(np.sin(angle_to_ball), np.cos(angle_to_ball)))
        current_angle_err = abs(angle_to_ball)

        if self.prev_angle_err is None:
            self.prev_angle_err = current_angle_err

        # Angular progress: positive if robot turned towards ball
        angle_closed = self.prev_angle_err - current_angle_err

        # Cosine alignment: +1.0 if facing ball directly, -1.0 if facing away
        alignment_score = float(np.cos(angle_to_ball))

        # 3. Dense Reward Formulation
        reward = (
                (distance_closed * 100.0)  # Distance progress
                + (alignment_score * 0.4)  # Step bonus for facing the ball
                + (angle_closed * 50.0)  # Reward for active turning towards ball
                - 1  # Step penalty to encourage speed
        )

        self.prev_distance = current_dist
        self.prev_angle_err = current_angle_err

        return reward

    def calculate_reward_after_touching(self, ball, striker):
        reward = 0.0
        if not striker.infrared:
            speed = (ball.vx_mps ** 2 * ball.vy_mps ** 2) ** 0.5
            if ball.vx_mps > 0:
                t = (GOAL_X - ball.x_m) / ball.vx_mps
                y_offset = min(0, (abs(ball.y_m + ball.vy_mps * t) - (0.5 * GOAL_SIZE)))
                if y_offset == 0:
                    reward = speed * 4 # ball will go in
                else:
                    reward = speed * (20 - y_offset * 2) # ball will not go in
            else:
                reward = 2.5 * ball.vx_mps # ball shot in wrong dir

        else:
            # look for optimal shooting angle
            angle_left = math.atan2(-0.5 * GOAL_SIZE - ball.y_m, GOAL_X - ball.x_m)
            angle_right = math.atan2(0.5 * GOAL_SIZE - ball.y_m, GOAL_X - ball.x_m)

            goal_angular_size = math.atan2(math.sin(angle_right - angle_left), math.cos(angle_right - angle_left))

            if self.prev_goal_angular_size is not None:
                reward = (goal_angular_size - self.prev_goal_angular_size) * 25 # goal angular size increased
            self.prev_goal_angular_size = goal_angular_size

        reward -= 1 # step penalty

        return reward