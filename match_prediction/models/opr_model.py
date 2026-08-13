from match_prediction.base_model import BaseModel
from match_prediction.context import MatchContext
from match_prediction.predictions import BinaryPrediction, Prediction, ScorePrediction


class OPRModel(BaseModel):
    claimed_outputs = [
        "winner",
        "red_alliance_score",
        "blue_alliance_score",
        "score_margin",
    ]

    def can_predict(self, context: MatchContext) -> bool:
        return len(context.team_stats) > 0

    def predict(self, context: MatchContext) -> dict[str, Prediction]:
        opr = context.global_context.get("opr", {})

        red = self._alliance_score(context.red_alliance, context.team_stats, opr)
        blue = self._alliance_score(context.blue_alliance, context.team_stats, opr)
        margin = red - blue

        # Logistic transform of score margin → win confidence
        import math
        red_win_conf = 1 / (1 + math.exp(-margin / 30))

        return {
            "winner": BinaryPrediction(
                predicted="red" if margin >= 0 else "blue",
                confidence=max(red_win_conf, 1 - red_win_conf),
            ),
            "red_alliance_score": ScorePrediction(mean=red),
            "blue_alliance_score": ScorePrediction(mean=blue),
            "score_margin": ScorePrediction(mean=margin),
        }

    def _alliance_score(self, teams, team_stats, opr) -> float:
        return sum(
            opr.get(t, team_stats[t].avg_total_score if t in team_stats else 0.0)
            for t in teams
        )
