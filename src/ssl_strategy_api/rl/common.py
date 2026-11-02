import math
import numpy as np
from ssl_strategy_api.models import RobotIntent

MAX_V, MAX_W, MAX_KICK = 2.0, 3.0, 5.0
OBS_DIM = 10


def build_obs(state, config, rid=0):
    r, b = state.robots_blue[rid], state.ball
    L, W = state.field.length_m, state.field.width_m
    h = r.heading_rad or 0.0
    c, s = math.cos(h), math.sin(h)

    def to_robot(dx, dy):                       # rotate field frame -> robot frame
        return c * dx + s * dy, -s * dx + c * dy

    gx = config.attack_direction_x * L / 2      # goal center x
    ball_rel = to_robot(b.x_m - r.x_m, b.y_m - r.y_m)
    goal_rel = to_robot(gx - r.x_m, -r.y_m)
    ball_vel = to_robot(b.vx_mps or 0.0, b.vy_mps or 0.0)
    rob_vel = to_robot(r.vx_mps or 0.0, r.vy_mps or 0.0)

    return np.array([
        ball_rel[0] / L, ball_rel[1] / W,
        goal_rel[0] / L, goal_rel[1] / W,
        ball_vel[0] / 5.0, ball_vel[1] / 5.0,
        rob_vel[0] / MAX_V, rob_vel[1] / MAX_V,
        (r.angular_velocity_rad_s or 0.0) / MAX_W,
        1.0 if r.infrared else 0.0,
    ], dtype=np.float32)


def action_to_intent(a, heading):
    a = np.clip(a, -1.0, 1.0)
    vxr, vyr = a[0] * MAX_V, a[1] * MAX_V       # in robot frame
    n = math.hypot(vxr, vyr)
    if n > MAX_V:                               # limit vector length, not components
        vxr, vyr = vxr * MAX_V / n, vyr * MAX_V / n
    c, s = math.cos(heading), math.sin(heading)
    return RobotIntent(                          # the API wants the field frame
        vx_mps=c * vxr - s * vyr,
        vy_mps=s * vxr + c * vyr,
        angular_velocity_rad_s=float(a[2]) * MAX_W,
        kick_speed_mps=MAX_KICK if a[3] > 0 else 0.0,
        dribbler=True,
    )
