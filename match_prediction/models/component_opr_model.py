from match_prediction.base_model import BaseModel
from match_prediction.context import MatchContext
from match_prediction.predictions import Prediction, ScorePrediction


class ComponentOPRModel(BaseModel):
    claimed_outputs = [
        "red_auto_score", "blue_auto_score",
        "red_teleop_score", "blue_teleop_score",
        "red_endgame_score", "blue_endgame_score",
        "red_alliance_score", "blue_alliance_score",
    ]

    def can_predict(self, context: MatchContext) -> bool:
        return len(context.team_stats) > 0

    def predict(self, context: MatchContext) -> dict[str, Prediction]:
        copr = context.global_context.get("component_opr", {})
        result: dict[str, Prediction] = {}

        for color in ("red", "blue"):
            teams = context.teams_for_alliance(color)
            auto    = self._sum(teams, context, copr, "auto",    "avg_auto_score")
            teleop  = self._sum(teams, context, copr, "teleop",  "avg_teleop_score")
            endgame = self._sum(teams, context, copr, "endgame", "avg_endgame_score")

            result[f"{color}_auto_score"]     = ScorePrediction(mean=auto)
            result[f"{color}_teleop_score"]   = ScorePrediction(mean=teleop)
            result[f"{color}_endgame_score"]  = ScorePrediction(mean=endgame)
            result[f"{color}_alliance_score"] = ScorePrediction(mean=auto + teleop + endgame)

        return result

    def _sum(self, teams, context, copr, component, fallback_attr) -> float:
        total = 0.0
        for team in teams:
            team_copr = copr.get(team, {})
            if component in team_copr:
                total += team_copr[component]
            elif team in context.team_stats:
                total += getattr(context.team_stats[team], fallback_attr, 0.0)
        return total
