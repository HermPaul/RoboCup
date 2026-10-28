from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Optional, Tuple


class TeamColor(str, Enum):
    BLUE = "blue"
    YELLOW = "yellow"


class VelocitySource(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


MAX_KICK_SPEED_MPS = 5.0
DEFAULT_KICK_SPEED_MPS = MAX_KICK_SPEED_MPS


def _validate_optional_finite(name: str, value: Optional[float]) -> None:
    if value is not None and not isfinite(value):
        raise ValueError("{} must be finite when provided".format(name))


@dataclass(frozen=True)
class FieldGeometry:
    length_m: float
    width_m: float
    goal_width_m: float

    def __post_init__(self) -> None:
        for name in ("length_m", "width_m", "goal_width_m"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError("{} must be positive and finite".format(name))


@dataclass(frozen=True)
class BallState:
    x_m: float
    y_m: float
    vx_mps: Optional[float] = None
    vy_mps: Optional[float] = None
    velocity_source: VelocitySource = VelocitySource.UNAVAILABLE

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m"):
            if not isfinite(getattr(self, name)):
                raise ValueError("{} must be finite".format(name))
        for name in ("vx_mps", "vy_mps"):
            _validate_optional_finite(name, getattr(self, name))
        if (self.vx_mps is None) != (self.vy_mps is None):
            raise ValueError("ball planar velocity must include both vx_mps and vy_mps")
        _validate_velocity_source(self.velocity_source, self.vx_mps, self.vy_mps)


@dataclass(frozen=True)
class RobotState:
    robot_id: int
    x_m: float
    y_m: float
    heading_rad: Optional[float] = None
    vx_mps: Optional[float] = None
    vy_mps: Optional[float] = None
    angular_velocity_rad_s: Optional[float] = None
    velocity_source: VelocitySource = VelocitySource.UNAVAILABLE
    infrared: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.robot_id < 0:
            raise ValueError("robot_id must be non-negative")
        for name in ("x_m", "y_m"):
            if not isfinite(getattr(self, name)):
                raise ValueError("{} must be finite".format(name))
        for name in ("heading_rad", "vx_mps", "vy_mps", "angular_velocity_rad_s"):
            _validate_optional_finite(name, getattr(self, name))
        if (self.vx_mps is None) != (self.vy_mps is None):
            raise ValueError("robot planar velocity must include both vx_mps and vy_mps")
        _validate_velocity_source(self.velocity_source, self.vx_mps, self.vy_mps)
def _validate_velocity_source(
    source: VelocitySource, vx_mps: Optional[float], vy_mps: Optional[float]
) -> None:
    if source == VelocitySource.UNAVAILABLE and vx_mps is not None:
        raise ValueError("unavailable velocity must not contain planar velocity values")
    if source != VelocitySource.UNAVAILABLE and vx_mps is None:
        raise ValueError("available velocity source requires planar velocity values")


@dataclass(frozen=True)
class WorldState:
    ball: Optional[BallState]
    robots_blue: Mapping[int, RobotState]
    robots_yellow: Mapping[int, RobotState]
    field: FieldGeometry
    timestamp_seconds: Optional[float] = None
    frame_number: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_optional_finite("timestamp_seconds", self.timestamp_seconds)
        for team_name in ("robots_blue", "robots_yellow"):
            robots = getattr(self, team_name)
            for robot_id, robot in robots.items():
                if robot_id != robot.robot_id:
                    raise ValueError("{} key must match RobotState.robot_id".format(team_name))


@dataclass(frozen=True)
class TeamConfig:
    team: TeamColor
    controlled_robot_ids: Tuple[int, ...]
    attack_direction_x: int

    def __post_init__(self) -> None:
        if self.attack_direction_x not in (-1, 1):
            raise ValueError("attack_direction_x must be -1 or 1")
        if len(set(self.controlled_robot_ids)) != len(self.controlled_robot_ids):
            raise ValueError("controlled_robot_ids must not contain duplicates")
        if any(robot_id < 0 for robot_id in self.controlled_robot_ids):
            raise ValueError("controlled robot ids must be non-negative")


@dataclass(frozen=True)
class RobotIntent:
    """Desired motion plus kick speed in m/s and a binary dribbler trigger."""

    vx_mps: float = 0.0
    vy_mps: float = 0.0
    angular_velocity_rad_s: float = 0.0
    kick_speed_mps: float = 0.0
    dribbler: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vx_mps",
            "vy_mps",
            "angular_velocity_rad_s",
            "kick_speed_mps",
        ):
            if not isfinite(getattr(self, name)):
                raise ValueError("{} must be finite".format(name))
        if not 0 <= self.kick_speed_mps <= MAX_KICK_SPEED_MPS:
            raise ValueError(
                "kick_speed_mps must be between 0 and {} m/s".format(
                    MAX_KICK_SPEED_MPS
                )
            )
        if not isinstance(self.dribbler, bool):
            raise ValueError("dribbler must be a boolean intent")


@dataclass(frozen=True)
class MotionLimits:
    max_linear_speed_mps: float = 2.0
    max_angular_speed_rad_s: float = 3.0

    def __post_init__(self) -> None:
        if not isfinite(self.max_linear_speed_mps) or not 0 <= self.max_linear_speed_mps <= 2.0:
            raise ValueError("max_linear_speed_mps must be between 0 and 2.0 m/s")
        if (
            not isfinite(self.max_angular_speed_rad_s)
            or not 0 <= self.max_angular_speed_rad_s <= 3.0
        ):
            raise ValueError("max_angular_speed_rad_s must be between 0 and 3 rad/s")


def clamp_intent(intent: RobotIntent, limits: MotionLimits) -> RobotIntent:
    """Clamp planar speed by vector magnitude and angular speed if configured."""
    speed = (intent.vx_mps ** 2 + intent.vy_mps ** 2) ** 0.5
    if speed > limits.max_linear_speed_mps and speed > 0:
        scale = limits.max_linear_speed_mps / speed
        vx_mps = intent.vx_mps * scale
        vy_mps = intent.vy_mps * scale
    else:
        vx_mps = intent.vx_mps
        vy_mps = intent.vy_mps

    bound = limits.max_angular_speed_rad_s
    angular_velocity = max(-bound, min(bound, intent.angular_velocity_rad_s))

    return RobotIntent(
        vx_mps=vx_mps,
        vy_mps=vy_mps,
        angular_velocity_rad_s=angular_velocity,
        kick_speed_mps=intent.kick_speed_mps,
        dribbler=intent.dribbler,
    )