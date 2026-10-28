from dataclasses import replace
from math import cos, isfinite, sin
from typing import Any, Dict, List, Mapping, Optional

from ssl_strategy_api.models import (
    BallState,
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


_MILLIMETERS_PER_METER = 1000.0


def field_geometry_from_team_control(field: Any) -> FieldGeometry:
    """Convert TeamControl SSL-Vision field dimensions from mm to meters."""
    return FieldGeometry(
        length_m=float(field.field_length) / _MILLIMETERS_PER_METER,
        width_m=float(field.field_width) / _MILLIMETERS_PER_METER,
        goal_width_m=float(field.goal_width) / _MILLIMETERS_PER_METER,
    )


def world_state_from_team_control_frame(
    frame: Any,
    field: FieldGeometry,
    timestamp_seconds: Optional[float] = None,
) -> WorldState:
    """Convert a TeamControl vision Frame from SSL mm coordinates to SI.

    SSL-Vision heading is already in radians. This codebase's Frame does not
    retain a capture timestamp or velocity, so this conversion marks velocity
    unavailable; pass the state to FrameVelocityEstimator when timestamps are
    available from the caller.
    """
    ball = None
    source_ball = getattr(frame, "ball", None)
    if source_ball is not None:
        x_m = _finite_float(getattr(source_ball, "x", None))
        y_m = _finite_float(getattr(source_ball, "y", None))
        if x_m is not None and y_m is not None:
            ball = BallState(x_m=x_m / 1000.0, y_m=y_m / 1000.0)

    blue = _robots_from_team_control(getattr(frame, "robots_blue", None))
    yellow = _robots_from_team_control(getattr(frame, "robots_yellow", None))
    return WorldState(
        ball=ball,
        robots_blue=blue,
        robots_yellow=yellow,
        field=field,
        timestamp_seconds=timestamp_seconds,
        frame_number=getattr(frame, "frame_number", None),
    )


def _robots_from_team_control(team: Any) -> Dict[int, RobotState]:
    converted: Dict[int, RobotState] = {}
    if team is None:
        return converted
    active_ids = getattr(team, "active", None)
    if active_ids is None:
        return converted
    for robot_id in active_ids:
        robot = team[robot_id]
        if robot is None or robot == 0:
            continue
        x_m = _finite_float(getattr(robot, "x", None))
        y_m = _finite_float(getattr(robot, "y", None))
        heading_rad = _finite_float(getattr(robot, "o", None))
        if x_m is None or y_m is None:
            continue
        converted[int(robot_id)] = RobotState(
            robot_id=int(robot_id),
            x_m=x_m / 1000.0,
            y_m=y_m / 1000.0,
            heading_rad=heading_rad,
        )
    return converted


class FrameVelocityEstimator:
    """Estimate planar velocities from successive timestamped world states."""

    def __init__(self) -> None:
        self._previous: Optional[WorldState] = None

    def reset(self) -> None:
        self._previous = None

    def update(self, state: WorldState) -> WorldState:
        timestamp = state.timestamp_seconds
        previous = self._previous
        self._previous = state if timestamp is not None else None
        if timestamp is None or previous is None or previous.timestamp_seconds is None:
            return state

        elapsed = timestamp - previous.timestamp_seconds
        if not isfinite(elapsed) or elapsed <= 0:
            return state

        ball = state.ball
        old_ball = previous.ball
        if ball is not None and old_ball is not None:
            ball = replace(
                ball,
                vx_mps=(ball.x_m - old_ball.x_m) / elapsed,
                vy_mps=(ball.y_m - old_ball.y_m) / elapsed,
                velocity_source=VelocitySource.ESTIMATED,
            )

        blue = _estimate_team_velocities(
            state.robots_blue, previous.robots_blue, elapsed
        )
        yellow = _estimate_team_velocities(
            state.robots_yellow, previous.robots_yellow, elapsed
        )
        return replace(state, ball=ball, robots_blue=blue, robots_yellow=yellow)


def _estimate_team_velocities(
    current: Mapping[int, RobotState],
    previous: Mapping[int, RobotState],
    elapsed: float,
) -> Dict[int, RobotState]:
    estimated: Dict[int, RobotState] = {}
    for robot_id, robot in current.items():
        old_robot = previous.get(robot_id)
        if old_robot is None:
            estimated[robot_id] = robot
            continue
        estimated[robot_id] = replace(
            robot,
            vx_mps=(robot.x_m - old_robot.x_m) / elapsed,
            vy_mps=(robot.y_m - old_robot.y_m) / elapsed,
            velocity_source=VelocitySource.ESTIMATED,
        )
    return estimated


def team_control_command_payloads(
    state: WorldState,
    config: TeamConfig,
    intents: Mapping[int, RobotIntent],
    limits: MotionLimits = MotionLimits(),
) -> List[Dict[str, Any]]:
    """Build TeamControl RobotCommand kwargs in grSim robot-relative axes."""
    controlled = set(config.controlled_robot_ids)
    unexpected = set(intents) - controlled
    if unexpected:
        raise ValueError("intents include uncontrolled robot ids: {}".format(sorted(unexpected)))
    robots = state.robots_blue if config.team == TeamColor.BLUE else state.robots_yellow
    payloads: List[Dict[str, Any]] = []
    for robot_id in config.controlled_robot_ids:
        intent = intents.get(robot_id)
        robot = robots.get(robot_id)
        if intent is None or robot is None or robot.heading_rad is None:
            continue
        bounded = clamp_intent(intent, limits)
        heading = robot.heading_rad
        vx_robot = bounded.vx_mps * cos(heading) + bounded.vy_mps * sin(heading)
        vy_robot = -bounded.vx_mps * sin(heading) + bounded.vy_mps * cos(heading)
        payloads.append(
            {
                "robot_id": robot_id,
                "vx": vx_robot,
                "vy": vy_robot,
                "w": bounded.angular_velocity_rad_s,
                "kick": int(bounded.kick_speed_mps > 0.0),
                "kick_speed_x": bounded.kick_speed_mps,
                "dribble": int(bounded.dribbler),
            }
        )
    return payloads


def to_team_control_commands(
    state: WorldState,
    config: TeamConfig,
    intents: Mapping[int, RobotIntent],
    limits: MotionLimits = MotionLimits(),
) -> List[Any]:
    """Convert portable intents to the existing TeamControl RobotCommand."""
    from TeamControl.network.robot_command import RobotCommand

    commands = []
    for payload in team_control_command_payloads(state, config, intents, limits):
        command_payload = {
            name: value for name, value in payload.items() if name != "kick_speed_x"
        }
        commands.append(RobotCommand(**command_payload))
    return commands


def _finite_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if isfinite(converted) else None