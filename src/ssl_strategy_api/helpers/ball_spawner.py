from dataclasses import dataclass
from math import hypot, isfinite
import random
from typing import Any, Optional, Tuple

from ssl_strategy_api.models import (
    BallState,
    FieldGeometry,
    TeamConfig,
    VelocitySource,
)


@dataclass(frozen=True)
class BallSpawn:
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float

    def to_state(self) -> BallState:
        return BallState(
            x_m=self.x_m,
            y_m=self.y_m,
            vx_mps=self.vx_mps,
            vy_mps=self.vy_mps,
            velocity_source=VelocitySource.MEASURED,
        )


class BallSpawner:
    """Spawn shots randomly in a team's defensive half, aimed at its goal."""

    def __init__(
        self,
        seed: Optional[int] = None,
        speed_range_mps: Tuple[float, float] = (4, 5),
        field_margin_m: float = 0.15,
        goal_target_margin_m: float = 0.08,
    ) -> None:
        minimum_speed, maximum_speed = speed_range_mps
        if (
            not isfinite(minimum_speed)
            or not isfinite(maximum_speed)
            or minimum_speed <= 0
            or maximum_speed < minimum_speed
        ):
            raise ValueError("speed_range_mps must be finite, positive, and ordered")
        if not isfinite(field_margin_m) or field_margin_m < 0:
            raise ValueError("field_margin_m must be non-negative and finite")
        if not isfinite(goal_target_margin_m) or goal_target_margin_m < 0:
            raise ValueError("goal_target_margin_m must be non-negative and finite")
        self._random = random.Random(seed)
        self.speed_range_mps = speed_range_mps
        self.field_margin_m = field_margin_m
        self.goal_target_margin_m = goal_target_margin_m

    def spawn_and_shoot_towards_goal(
        self,
        field: FieldGeometry,
        config: TeamConfig,
    ) -> BallSpawn:
        defense_direction = -config.attack_direction_x
        half_length = field.length_m / 2
        half_width = field.width_m / 2
        x_limit = max(0.0, half_length - self.field_margin_m)
        y_limit = max(0.0, half_width - self.field_margin_m)
        if defense_direction < 0:
            x_m = self._random.uniform(-x_limit, 0.0)
        else:
            x_m = self._random.uniform(0.0, x_limit)
        y_m = self._random.uniform(-y_limit, y_limit)

        goal_y_limit = max(0.0, field.goal_width_m / 2 - self.goal_target_margin_m)
        target_y = self._random.uniform(-goal_y_limit, goal_y_limit)
        target_x = defense_direction * half_length
        dx = target_x - x_m
        dy = target_y - y_m
        distance = hypot(dx, dy)
        speed = self._random.uniform(*self.speed_range_mps)
        return BallSpawn(
            x_m=x_m,
            y_m=y_m,
            vx_mps=speed * dx / distance,
            vy_mps=speed * dy / distance,
        )

    def spawn_stationary_right_side(
        self,
        field: Any,
        attack_direction_x: int,
        right_side_margin_m: float = 0.20,
        enforce_outside_opponent_goal_area: bool = True,
    ) -> BallSpawn:
        """Spawn a stationary point on the right half, optionally outside opponent goal area."""
        if attack_direction_x not in (-1, 1):
            raise ValueError("attack_direction_x must be -1 or 1")
        if not isfinite(right_side_margin_m) or right_side_margin_m < 0:
            raise ValueError("right_side_margin_m must be non-negative and finite")

        length_m = _field_value(field, "length_m", "length")
        width_m = _field_value(field, "width_m", "width")
        half_length = length_m / 2
        half_width = width_m / 2
        x_min = right_side_margin_m
        x_max = max(x_min, half_length - right_side_margin_m)
        y_min = -half_width + right_side_margin_m
        y_max = half_width - right_side_margin_m

        
        x_m = self._random.uniform(x_min, x_max)
        if not enforce_outside_opponent_goal_area:
            y_m = self._random.uniform(y_min, y_max)
            return BallSpawn(x_m=x_m, y_m=y_m, vx_mps=0.0, vy_mps=0.0)
        
        """ 
        # JUST TO TEST REPOSITIONING
        x_m = 4
        if not enforce_outside_opponent_goal_area:
            y_m = 2.5
            return BallSpawn(x_m=x_m, y_m=y_m, vx_mps=0.0, vy_mps=0.0) """

        if not self.in_opponent_goal_area(field, x_m, 0.0, attack_direction_x):
            y_m = self._random.uniform(y_min, y_max)
            return BallSpawn(x_m=x_m, y_m=y_m, vx_mps=0.0, vy_mps=0.0)

        y_m = self._sample_y_outside_opponent_goal_band(
            field,
            y_min,
            y_max,
        )
        if y_m is not None:
            return BallSpawn(x_m=x_m, y_m=y_m, vx_mps=0.0, vy_mps=0.0)

        # Degenerate-field fallback: move x away from the opponent goal-area x-band.
        penalty_length_m = _field_value(
            field,
            "penalty_length_m",
            "penalty_length",
        )
        opponent_goal_area_x_min = half_length - penalty_length_m
        safe_x_max = min(x_max, opponent_goal_area_x_min - 1e-6)
        if safe_x_max > x_min:
            x_m = self._random.uniform(x_min, safe_x_max)
        y_m = self._random.uniform(y_min, y_max)
        return BallSpawn(x_m=x_m, y_m=y_m, vx_mps=0.0, vy_mps=0.0)

    def respawn_when_idle_or_out(
        self,
        ball: Optional[BallState],
        field: FieldGeometry,
        stopped_speed_mps: float = 0.08,
        out_of_bounds_margin_m: float = 0.05,
    ) -> Optional[str]:
        if ball is None:
            return None
        if (
            abs(ball.x_m) > field.length_m / 2 + out_of_bounds_margin_m
            or abs(ball.y_m) > field.width_m / 2 + out_of_bounds_margin_m
        ):
            return "out-of-bounds"
        if ball.vx_mps is not None and ball.vy_mps is not None:
            if hypot(ball.vx_mps, ball.vy_mps) <= stopped_speed_mps:
                return "stopped"
        return None

    def respawn_when_goal_or_out_or_idle(
        self,
        ball: Optional[BallState],
        field: Any,
        attack_direction_x: int,
        idle_speed_mps: float = 0.06,
    ) -> Optional[str]:
        """Return respawn reason for exercise1-style restarts.

        Triggers on:
        - any goal scored
        - out-of-bounds
        - ball idle while inside the opponent goal area
        """
        if attack_direction_x not in (-1, 1):
            raise ValueError("attack_direction_x must be -1 or 1")
        if not isfinite(idle_speed_mps) or idle_speed_mps < 0:
            raise ValueError("idle_speed_mps must be non-negative and finite")
        if ball is None:
            return None

        length_m = _field_value(field, "length_m", "length")
        width_m = _field_value(field, "width_m", "width")
        goal_width_m = _field_value(field, "goal_width_m", "goal_width")
        goal_depth_m = _field_value(field, "goal_depth_m", "goal_depth", default=0.0)
        half_length = length_m / 2
        half_width = width_m / 2
        half_goal = goal_width_m / 2
        if (
            (ball.x_m >= half_length and abs(ball.y_m) <= half_goal)
            or (ball.x_m <= -half_length and abs(ball.y_m) <= half_goal)
        ):
            return "goal"

        if abs(ball.y_m) > half_width or abs(ball.x_m) > half_length + goal_depth_m:
            return "out-of-bounds"

        if (
            self.in_opponent_goal_area(field, ball.x_m, ball.y_m, attack_direction_x)
            and ball.vx_mps is not None
            and ball.vy_mps is not None
            and hypot(ball.vx_mps, ball.vy_mps) <= idle_speed_mps
        ):
            return "idle-in-opponent-goal-area"
        return None

    def in_opponent_goal_area(
        self,
        field: Any,
        x_m: float,
        y_m: float,
        attack_direction_x: int,
    ) -> bool:
        if attack_direction_x not in (-1, 1):
            raise ValueError("attack_direction_x must be -1 or 1")
        length_m = _field_value(field, "length_m", "length")
        penalty_length_m = _field_value(
            field,
            "penalty_length_m",
            "penalty_length",
        )
        penalty_width_m = _field_value(
            field,
            "penalty_width_m",
            "penalty_width",
        )
        half_length = length_m / 2
        half_penalty_width = penalty_width_m / 2
        if attack_direction_x > 0:
            x_limit = half_length - penalty_length_m
            return x_m >= x_limit and abs(y_m) <= half_penalty_width
        x_limit = -half_length + penalty_length_m
        return x_m <= x_limit and abs(y_m) <= half_penalty_width

    def _sample_y_outside_opponent_goal_band(
        self,
        field: Any,
        y_min: float,
        y_max: float,
    ) -> Optional[float]:
        penalty_width_m = _field_value(field, "penalty_width_m", "penalty_width")
        half_penalty_width = penalty_width_m / 2

        lower_start = y_min
        lower_end = min(y_max, -half_penalty_width)
        upper_start = max(y_min, half_penalty_width)
        upper_end = y_max

        lower_len = max(0.0, lower_end - lower_start)
        upper_len = max(0.0, upper_end - upper_start)
        total_len = lower_len + upper_len
        if total_len <= 0.0:
            return None

        pick = self._random.uniform(0.0, total_len)
        if pick < lower_len:
            return self._random.uniform(lower_start, lower_end)
        return self._random.uniform(upper_start, upper_end)


def _field_value(field: Any, primary: str, fallback: str, default: Optional[float] = None) -> float:
    value = getattr(field, primary, None)
    if value is None:
        value = getattr(field, fallback, None)
    if value is None:
        if default is None:
            raise ValueError("field is missing required attribute {} or {}".format(primary, fallback))
        value = default
    converted = float(value)
    if not isfinite(converted):
        raise ValueError("field attribute {} must be finite".format(primary))
    return converted