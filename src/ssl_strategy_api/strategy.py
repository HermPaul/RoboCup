from typing import Mapping, Optional, Protocol

from ssl_strategy_api.models import RobotIntent, TeamConfig, WorldState


class Strategy(Protocol):
    def decide(
        self, state: WorldState, config: TeamConfig
    ) -> Mapping[int, RobotIntent]:
        """Return one physical command intent per controlled robot id."""


class ResettableStrategy(Strategy, Protocol):
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset episode-local strategy state."""