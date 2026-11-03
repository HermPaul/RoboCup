from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ssl_strategy_api.run_demos.run_rsoccer_training import TrainingEnv

BLUE_ROBOT_ID = 0
BALL_IDLE_SPEED_MPS = 0.06


class TouchEvaluator:
    """Evaluates per-step dense rewards and finish criteria."""

    def __init__(self):
        self.prev_angle_err: Optional[float] = None
        self.prev_distance: Optional[float] = None
        self.already_touched: bool = False

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

        goal_x = (env.field.goal_width / 2.0) * env.blue_config.attack_direction_x
        goal_y = 0.0

        bg_dx = goal_x - ball.x_m
        bg_dy = goal_y - ball.y_m
        bg_dist = float(np.hypot(bg_dx, bg_dy))

        rb_dx = striker.x_m - ball.x_m
        rb_dy = ball.y_m - striker.y_m
        rb_dist = float(np.hypot(rb_dx, rb_dy))

        if rb_dist > 1e-5 and bg_dist > 1e-5:
            #goal_alignment_score = float((rb_dx * bg_dx + rb_dy * bg_dy) / (rb_dist * bg_dist))
            dot = (rb_dx * bg_dx + rb_dy * bg_dy) / (rb_dist * bg_dist)

            # Reward being behind the ball relative to the opponent goal
            goal_alignment_score = max(0.0, -dot)
        else:
            goal_alignment_score = 0.0

        # Contact / Touch handling (Fallback to close proximity if infrared is finicky)
        ball_touched = bool(striker.infrared) or (rb_dist < 0.15)
        if ball_touched and not self.already_touched:
            reward += 10.0  # One-time touch bonus
            self.already_touched = True

        # Ball velocity toward opponent goal
        ball_vel_toward_goal = ball.vx_mps * env.blue_config.attack_direction_x
        ball_speed = float(np.hypot(ball.vx_mps, ball.vy_mps))
        
        # Reward ball moving forward regardless of strict infrared state
        propulsion_reward = (ball_speed * 5.0) if (rb_dist < 0.2) else 0.0

        # 3. Dense Reward Formulation (Softened step penalty)
        dense_reward = (
            (distance_closed * 50.0)        # Distance progress
            + (alignment_score * 0.2)       # Facing the ball
            + (angle_closed * 10.0)         # Turning toward the ball
            + (max(0.0, goal_alignment_score) * 2.0) # Positioning behind ball
            + (max(0.0, ball_vel_toward_goal) * 5.0) # Pushing ball toward goal
            + propulsion_reward             # Striking/propelling reward
            - 0.01                          # MILD step penalty (prevents freezing instead of -1.0)
        )
        reward += dense_reward

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
        
        if respawn_reason == "goal":
            reward += 1000.0 
            finish_reason = "goal_scored"
            done = True
        elif blue_in_opponent_goal_area:
            done = True
            reward -= 10.0
            finish_reason = "goal_area_foul"
        elif respawn_reason is not None:
            done = True
            reward -= 10.0
            finish_reason = f"out_or_idle_{respawn_reason}"

        if done:
            self.prev_distance = None
            self.prev_angle_err = None
            self.already_touched = False
        else:
            self.prev_distance = current_dist
            self.prev_angle_err = current_angle_err

        info = {
            "finish_reason": finish_reason,
            "touched_ball": ball_touched,
        }
        return reward, done, info