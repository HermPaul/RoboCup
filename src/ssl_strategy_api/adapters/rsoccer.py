from math import cos, isfinite, radians, sin
from typing import Any, Dict, List, Mapping, Optional

from ssl_strategy_api.models import (
    BallState,
    DEFAULT_KICK_SPEED_MPS,
    FieldGeometry,
    MotionLimits,
    RobotIntent,
    RobotState,
    TeamConfig,
    TeamColor,
    VelocitySource,
    WorldState,
    clamp_intent,
)


def _finite_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if isfinite(converted) else None


def world_state_from_rsoccer_frame(
    frame: Any,
    field: FieldGeometry,
    timestamp_seconds: Optional[float] = None,
    frame_number: Optional[int] = None,
) -> WorldState:
    """Convert an rSoccer SSL Frame into canonical SI units.

    rSoccer positions and linear velocities already use meters and m/s. Its
    theta and v_theta fields use degrees and degrees/s, converted here to
    radians and rad/s. Missing or non-finite detections are omitted.
    """
    ball = None
    source_ball = getattr(frame, "ball", None)
    if source_ball is not None:
        x_m = _finite_float(getattr(source_ball, "x", None))
        y_m = _finite_float(getattr(source_ball, "y", None))
        if x_m is not None and y_m is not None:
            vx_mps = _finite_float(getattr(source_ball, "v_x", None))
            vy_mps = _finite_float(getattr(source_ball, "v_y", None))
            if vx_mps is None or vy_mps is None:
                vx_mps = None
                vy_mps = None
                velocity_source = VelocitySource.UNAVAILABLE
            else:
                velocity_source = VelocitySource.MEASURED
            ball = BallState(
                x_m=x_m,
                y_m=y_m,
                vx_mps=vx_mps,
                vy_mps=vy_mps,
                velocity_source=velocity_source,
            )

    blue = _robots_from_rsoccer(getattr(frame, "robots_blue", {}))
    yellow = _robots_from_rsoccer(getattr(frame, "robots_yellow", {}))
    return WorldState(
        ball=ball,
        robots_blue=blue,
        robots_yellow=yellow,
        field=field,
        timestamp_seconds=timestamp_seconds,
        frame_number=frame_number,
    )


def _robots_from_rsoccer(robots: Mapping[int, Any]) -> Dict[int, RobotState]:
    converted: Dict[int, RobotState] = {}
    for key, robot in robots.items():
        if robot is None:
            continue
        robot_id_value = getattr(robot, "id", key)
        try:
            robot_id = int(robot_id_value)
        except (TypeError, ValueError):
            continue
        x_m = _finite_float(getattr(robot, "x", None))
        y_m = _finite_float(getattr(robot, "y", None))
        if x_m is None or y_m is None:
            continue

        vx_mps = _finite_float(getattr(robot, "v_x", None))
        vy_mps = _finite_float(getattr(robot, "v_y", None))
        if vx_mps is None or vy_mps is None:
            vx_mps = None
            vy_mps = None
            velocity_source = VelocitySource.UNAVAILABLE
        else:
            velocity_source = VelocitySource.MEASURED

        theta_degrees = _finite_float(getattr(robot, "theta", None))
        omega_degrees = _finite_float(getattr(robot, "v_theta", None))
        infrared = getattr(robot, "infrared", None)
        if infrared is not None:
            infrared = bool(infrared)

        converted[robot_id] = RobotState(
            robot_id=robot_id,
            x_m=x_m,
            y_m=y_m,
            heading_rad=radians(theta_degrees) if theta_degrees is not None else None,
            vx_mps=vx_mps,
            vy_mps=vy_mps,
            angular_velocity_rad_s=(
                radians(omega_degrees) if omega_degrees is not None else None
            ),
            velocity_source=velocity_source,
            infrared=infrared,
        )
    return converted


def rsoccer_command_payloads(
    state: WorldState,
    config: TeamConfig,
    intents: Mapping[int, RobotIntent],
    limits: MotionLimits = MotionLimits(),
) -> List[Dict[str, Any]]:
    """Build rSoccer Robot constructor kwargs without importing rSoccer.

    Strategy velocities are expressed in field coordinates. rSoccer SSL
    commands expect robot-relative planar velocity, so this applies the
    inverse heading rotation. Angular commands use rad/s, unlike the
    degree-based angular fields in observed frames.
    Kick speed is carried by the portable RobotIntent.
    """
    controlled = set(config.controlled_robot_ids)
    unexpected = set(intents) - controlled
    if unexpected:
        raise ValueError("intents include uncontrolled robot ids: {}".format(sorted(unexpected)))

    team_robots = (
        state.robots_blue if config.team == TeamColor.BLUE else state.robots_yellow
    )
    payloads: List[Dict[str, Any]] = []
    for robot_id in config.controlled_robot_ids:
        intent = intents.get(robot_id)
        robot = team_robots.get(robot_id)
        if intent is None or robot is None or robot.heading_rad is None:
            continue

        bounded = clamp_intent(intent, limits)
        heading = robot.heading_rad
        vx_robot = bounded.vx_mps * cos(heading) + bounded.vy_mps * sin(heading)
        vy_robot = -bounded.vx_mps * sin(heading) + bounded.vy_mps * cos(heading)
        payloads.append(
            {
                "yellow": config.team == TeamColor.YELLOW,
                "id": robot_id,
                "v_x": vx_robot,
                "v_y": vy_robot,
                "v_theta": bounded.angular_velocity_rad_s,
                "kick_v_x": bounded.kick_speed_mps,
                "dribbler": bounded.dribbler,
            }
        )
    return payloads


def to_rsoccer_commands(
    state: WorldState,
    config: TeamConfig,
    intents: Mapping[int, RobotIntent],
    limits: MotionLimits = MotionLimits(),
) -> List[Any]:
    """Convert portable intents into rSoccer SSL Robot commands."""
    from rsoccer_gym.Entities import Robot

    return [
        Robot(**payload)
        for payload in rsoccer_command_payloads(
            state, config, intents, limits=limits
        )
    ]