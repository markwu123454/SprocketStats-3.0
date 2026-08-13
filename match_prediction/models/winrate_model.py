from match_prediction.base_model import BaseModel
from match_prediction.context import MatchContext
from match_prediction.predictions import BinaryPrediction, Prediction


class WinRateModel(BaseModel):
    claimed_outputs = ["winner"]

    def can_predict(self, context: MatchContext) -> bool:
        return (
            len(context.stats_for_alliance("red")) > 0
            and len(context.stats_for_alliance("blue")) > 0
        )

    def predict(self, context: MatchContext) -> dict[str, Prediction]:
        red_wr = sum(s.win_rate for s in context.stats_for_alliance("red"))
        blue_wr = sum(s.win_rate for s in context.stats_for_alliance("blue"))

        total = red_wr + blue_wr
        red_prob = red_wr / total if total > 0 else 0.5

        return {
            "winner": BinaryPrediction(
                predicted="red" if red_prob >= 0.5 else "blue",
                confidence=max(red_prob, 1 - red_prob),
            )
        }
