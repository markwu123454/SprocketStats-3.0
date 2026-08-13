import math

from match_prediction.base_model import BaseModel
from match_prediction.context import MatchContext
from match_prediction.predictions import Prediction, ScorePrediction


class EPAModel(BaseModel):
    claimed_outputs = [
        "robot_auto_contribution",
        "robot_teleop_contribution",
        "robot_endgame_contribution",
        "robot_total_contribution",
    ]

    def can_predict(self, context: MatchContext) -> bool:
        return any(
            t in context.team_stats
            for t in context.red_alliance + context.blue_alliance
        )

    def predict(self, context: MatchContext) -> dict[str, Prediction]:
        result: dict[str, Prediction] = {}
        for team in context.red_alliance + context.blue_alliance:
            if team not in context.team_stats:
                continue
            s = context.team_stats[team]
            stdev = math.sqrt(s.score_variance)
            result[f"robot_auto_contribution_{team}"] = ScorePrediction(s.avg_auto_score, stdev * 0.4)
            result[f"robot_teleop_contribution_{team}"] = ScorePrediction(s.avg_teleop_score, stdev * 0.7)
            result[f"robot_endgame_contribution_{team}"] = ScorePrediction(s.avg_endgame_score, stdev * 0.3)
            result[f"robot_total_contribution_{team}"] = ScorePrediction(s.avg_total_score, stdev)
        return result
