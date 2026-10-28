from typing import Dict, Optional

from ssl_strategy_api.models import BallState, FieldGeometry, TeamColor, TeamConfig


class GoalTracker:
    """Detect goal-line crossings and maintain a shared Blue-vs-Yellow score."""

    def __init__(self) -> None:
        self.scores: Dict[TeamColor, int] = {
            TeamColor.BLUE: 0,
            TeamColor.YELLOW: 0,
        }
        self._previous_ball: Optional[BallState] = None

    @property
    def total_goals(self) -> int:
        return self.scores[TeamColor.BLUE] + self.scores[TeamColor.YELLOW]

    def reset_ball_tracking(self, ball: Optional[BallState] = None) -> None:
        self._previous_ball = ball

    def update(
        self,
        ball: Optional[BallState],
        field: FieldGeometry,
        config: TeamConfig,
    ) -> Optional[TeamColor]:
        if ball is None:
            self._previous_ball = None
            return None

        previous_ball = self._previous_ball
        self._previous_ball = ball
        if previous_ball is None:
            return None

        half_length = field.length_m / 2
        if previous_ball.x_m < half_length <= ball.x_m:
            goal_side = 1
        elif previous_ball.x_m > -half_length >= ball.x_m:
            goal_side = -1
        else:
            return None

        if abs(ball.y_m) > field.goal_width_m / 2:
            return None

        scorer = (
            config.team
            if config.attack_direction_x == goal_side
            else _opponent(config.team)
        )
        self.scores[scorer] += 1
        print("{} Team scored a goal.".format(scorer.value.title()), flush=True)
        print(self.score_line, flush=True)
        return scorer

    @property
    def score_line(self) -> str:
        return "Blue Team {} - Yellow Team {}".format(
            self.scores[TeamColor.BLUE], self.scores[TeamColor.YELLOW]
        )


def _opponent(team: TeamColor) -> TeamColor:
    return TeamColor.YELLOW if team == TeamColor.BLUE else TeamColor.BLUE