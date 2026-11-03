import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ssl_strategy_api.run_demos.run_rsoccer_training import TrainingEnv

BLUE_ROBOT_ID = 0
BALL_IDLE_SPEED_MPS = 0.06
GOAL_X = 4.5
GOAL_Y = 0.0


class TouchAndShootEvaluator:
    """Evaluates per-step dense rewards and finish criteria for shooting tasks."""

    def __init__(self):
        self.prev_angle_err: Optional[float] = None
        self.prev_distance: Optional[float] = None
        self.ball_already_touched: bool = False
        self.prev_d_ball: Optional[float] = None

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
            reward -= 2000.0
            finish_reason = "goal_area_foul"
        elif respawn_reason == "goal":
            done = True
            reward += 5000.0
            finish_reason = "goal"
        elif respawn_reason is not None:
            done = True
            reward -= 50.0
            finish_reason = f"out_or_idle_{respawn_reason}"
        elif env.steps >= env.max_steps:
            done = True
            finish_reason = "max_duration_reached"

        if done:
            self.prev_distance = None
            self.prev_angle_err = None
            self.ball_already_touched = False
            self.prev_d_ball = None

        info = {
            "finish_reason": finish_reason,
            "touched_ball": self.ball_already_touched,
        }
        return reward, done, info

    def calculate_reward_before_touching(self, ball, striker) -> float:
        dx = ball.x_m - striker.x_m
        dy = ball.y_m - striker.y_m
        current_dist = float(np.hypot(dx, dy))

        if self.prev_distance is None:
            self.prev_distance = current_dist

        distance_closed = self.prev_distance - current_dist

        angle_to_ball = float(np.arctan2(dy, dx) - striker.heading_rad)
        angle_to_ball = float(np.arctan2(np.sin(angle_to_ball), np.cos(angle_to_ball)))
        current_angle_err = abs(angle_to_ball)

        if self.prev_angle_err is None:
            self.prev_angle_err = current_angle_err

        angle_closed = self.prev_angle_err - current_angle_err
        alignment_score = float(np.cos(angle_to_ball))

        reward = (
            (distance_closed * 100.0)
            + (alignment_score * 0.4)
            + (angle_closed * 50.0)
            - 0.02
        )

        self.prev_distance = current_dist
        self.prev_angle_err = current_angle_err

        return reward

    def calculate_reward_after_touching(self, ball, striker) -> float:
        ball_pos = (ball.x_m, ball.y_m)
        d_ball_goal = math.hypot(GOAL_X - ball.x_m, GOAL_Y - ball.y_m)

        ball_progress = 0.0
        if self.prev_d_ball is None:
            self.prev_d_ball = d_ball_goal
        else:
            ball_progress = self.prev_d_ball - d_ball_goal

        d_robot_ball = math.hypot(ball_pos[0] - striker.x_m, ball_pos[1] - striker.y_m)
        has_possession = bool(getattr(striker, "infrared", False)) or (d_robot_ball <= 0.12)

        # 1. Goal Alignment
        dx_goal = GOAL_X - striker.x_m
        dy_goal = GOAL_Y - striker.y_m
        angle_to_goal = math.atan2(dy_goal, dx_goal) - striker.heading_rad
        angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
        goal_facing_score = math.cos(angle_to_goal)

        # 2. Projection of Ball Velocity Vector toward Goal
        v_bg = (GOAL_X - ball.x_m, GOAL_Y - ball.y_m)
        mag_bg = math.hypot(*v_bg) + 1e-6
        unit_v_bg = (v_bg[0] / mag_bg, v_bg[1] / mag_bg)
        v_ball_toward_goal = ball.vx_mps * unit_v_bg[0] + ball.vy_mps * unit_v_bg[1]

        # 3. Detect Active Shot in Flight
        is_shot_in_flight = (v_ball_toward_goal > 1.2) and (d_ball_goal < 4.5)

        # 4. Possession vs. Flight State Incentive
        if has_possession:
            possession_score = 0.5
        elif is_shot_in_flight:
            possession_score = 1.0  # Bonus for a shot traveling toward the goal
        else:
            possession_score = -0.5  # Mild penalty for losing control off-target

        reward = (
            (ball_progress * 200.0)
            + (v_ball_toward_goal * 50.0)  # Strong incentive for high-speed kicks
            + possession_score
            + (goal_facing_score * 0.4)
            - 0.02
        )

        self.prev_d_ball = d_ball_goal

        return reward