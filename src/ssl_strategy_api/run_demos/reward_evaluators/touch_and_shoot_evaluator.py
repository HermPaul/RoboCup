import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ssl_strategy_api.run_demos.run_rsoccer_training import TrainingEnv

BLUE_ROBOT_ID = 0
BALL_IDLE_SPEED_MPS = 0.06
GOAL_X = 4.5
GOAL_Y = 0.0
GOAL_SIZE = 1.0


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
            reward -= 100.0
            finish_reason = "foul_in_goal_area"
        elif respawn_reason == "goal":
            done = True
            if ball.x_m > 0:
                reward += 1500.0  # Huge bonus for scoring in the opponent's goal
                finish_reason = "goal"
            else:
                reward -= 100.0   # Penalty for own goal
                finish_reason = "own_goal"
        elif respawn_reason is not None:
            done = True
            reward -= 50.0        # Out of bounds or stopped in box
            finish_reason = f"out_or_idle_{respawn_reason}"

        # Save touched status BEFORE resetting for the next episode
        touched_this_episode = self.ball_already_touched

        if done:
            self.prev_distance = None
            self.prev_angle_err = None
            self.ball_already_touched = False
            self.prev_goal_angular_size = None

        info = {
            "finish_reason": finish_reason,
            "touched_ball": touched_this_episode,
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
        angle_to_ball = float(np.arctan2(np.sin(angle_to_ball), np.cos(angle_to_ball)))
        current_angle_err = abs(angle_to_ball)

        if self.prev_angle_err is None:
            self.prev_angle_err = current_angle_err

        angle_closed = self.prev_angle_err - current_angle_err
        alignment_score = float(np.cos(angle_to_ball))

        # 3. Dense Reward Formulation
        reward = (
            (distance_closed * 100.0)
            + (alignment_score * 0.4)
            + (angle_closed * 50.0)
            - 1.0
        )

        self.prev_distance = current_dist
        self.prev_angle_err = current_angle_err

        return reward

    def calculate_reward_after_touching(self, ball, striker):
            # Always keep prev_distance and prev_angle_err updated so Case 3 never gets a huge jump penalty!
            dx = ball.x_m - striker.x_m
            dy = ball.y_m - striker.y_m
            current_dist = float(np.hypot(dx, dy))
            angle_to_ball = float(np.arctan2(dy, dx) - striker.heading_rad)
            angle_to_ball = float(np.arctan2(np.sin(angle_to_ball), np.cos(angle_to_ball)))
            current_angle_err = abs(angle_to_ball)

            vx = ball.vx_mps or 0.0
            vy = ball.vy_mps or 0.0
            speed = math.hypot(vx, vy)

            # Case 1: Robot currently has the ball on its dribbler/kicker
            if striker.infrared:
                self.prev_distance = current_dist
                self.prev_angle_err = current_angle_err
                dx_goal = GOAL_X - striker.x_m
                dy_goal = GOAL_Y - striker.y_m
                angle_to_goal = math.atan2(dy_goal, dx_goal) - striker.heading_rad
                angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
                return 3.0 * math.cos(angle_to_goal) + 1.0

            # Case 2: Ball is moving toward the opponent goal (vx > 0.2)
            if vx > 0.2:
                self.prev_distance = current_dist
                self.prev_angle_err = current_angle_err
                t = (GOAL_X - ball.x_m) / vx
                pred_y = ball.y_m + vy * t
                y_miss = max(0.0, abs(pred_y) - (0.5 * GOAL_SIZE))
                if y_miss == 0.0:
                    return speed * 15.0  # Strong reward for shot on target
                else:
                    return speed * max(0.0, 4.0 - y_miss * 3.0) - 0.2

            # Case 3: Ball stopped or went backward -> chase it again (without punishing ball roll-away)
            prev_d = self.prev_distance if self.prev_distance is not None else current_dist
            prev_a = self.prev_angle_err if self.prev_angle_err is not None else current_angle_err
            distance_closed = max(0.0, prev_d - current_dist)  # Only reward closing distance, never punish ball moving away
            angle_closed = prev_a - current_angle_err

            self.prev_distance = current_dist
            self.prev_angle_err = current_angle_err

            return (distance_closed * 100.0) + (float(np.cos(angle_to_ball)) * 0.4) + (angle_closed * 50.0) - 0.5