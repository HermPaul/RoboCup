from typing import Mapping

from ssl_strategy_api.models import RobotIntent, TeamConfig, WorldState


class IdleGoalkeeperStrategy:
    """Simple no-op goalkeeper strategy.

    This keeps the first controlled robot in place by returning a zero-velocity intent.
    """

    def reset(self, seed=None) -> None:
        pass

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids:
            return {}

        robot_id = config.controlled_robot_ids[0]
        return {
            robot_id: RobotIntent(
                vx_mps=0.0,
                vy_mps=0.0,
                angular_velocity_rad_s=0.0,
            )
        }
