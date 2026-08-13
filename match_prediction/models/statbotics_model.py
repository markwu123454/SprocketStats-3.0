"""
StatboticsModel — two distinct fetch paths, each used where it's temporally clean.

PATH 1 — get_match(match_key):
  Statbotics pre-computes alliance EPA sums and win probability at match time
  using the EPA snapshot *before* that match was played. No leakage, even on
  historical matches. This drives winner, alliance scores, component scores.

PATH 2 — get_team_event() with *_start fields only:
  epa_start / auto_epa_start / etc. are the pre-event EPA baseline — computed
  before the event began. We never use epa_end or epa_mean because those are
  aggregated over all event matches, so using them to predict any specific match
  would leak future results into historical weight calibration.

Why not use epa_end for live prediction?
  Even for a real-time upcoming match, epa_end includes all matches played so
  far in the event. That makes the model's predictions look better than they
  should on past matches when we evaluate weights — the weight tracker sees
  an inflated accuracy, messing up the relative weighting against other models.
"""

import logging
import math
from typing import Optional

try:
    import statbotics as _sb_lib
    _STATBOTICS_AVAILABLE = True
except ImportError:
    _STATBOTICS_AVAILABLE = False

from match_prediction.base_model import BaseModel
from match_prediction.context import MatchContext
from match_prediction.predictions import BinaryPrediction, Prediction, ScorePrediction

logger = logging.getLogger(__name__)


def _strip_frc(team_key: str) -> int:
    return int(team_key.replace("frc", ""))


class StatboticsModel(BaseModel):
    claimed_outputs = [
        # From get_match — temporally clean
        "winner",
        "red_alliance_score",
        "blue_alliance_score",
        "red_auto_score",
        "blue_auto_score",
        "red_teleop_score",
        "blue_teleop_score",
        "red_endgame_score",
        "blue_endgame_score",
        # From get_team_event epa_start — pre-event baseline, no leakage
        "robot_auto_contribution",
        "robot_teleop_contribution",
        "robot_endgame_contribution",
        "robot_total_contribution",
    ]

    def __init__(self):
        if not _STATBOTICS_AVAILABLE:
            raise ImportError("pip install statbotics")
        self._client = _sb_lib.Statbotics()
        self._match_cache: dict[str, Optional[dict]] = {}
        self._team_event_cache: dict[tuple[str, str], Optional[dict]] = {}

    def can_predict(self, context: MatchContext) -> bool:
        return True

    def predict(self, context: MatchContext) -> dict[str, Prediction]:
        result: dict[str, Prediction] = {}

        # ── Path 1: match-level prediction from get_match ────────────────────
        match_data = self._fetch_match(context.match_key)
        if match_data:
            result.update(self._predictions_from_match(match_data))

        # ── Path 2: per-robot from pre-event EPA baseline ────────────────────
        for team_key in context.red_alliance + context.blue_alliance:
            te = self._fetch_team_event(team_key, context.event_key)
            if te:
                result.update(self._robot_predictions(team_key, te))

        return result

    # ── Match-level predictions ───────────────────────────────────────────────

    def _predictions_from_match(self, m: dict) -> dict[str, Prediction]:
        out: dict[str, Prediction] = {}

        red_epa  = m.get("red_epa_sum")
        blue_epa = m.get("blue_epa_sum")
        win_prob = m.get("epa_win_prob")  # probability that red wins

        if red_epa is not None:
            out["red_alliance_score"] = ScorePrediction(mean=red_epa)
        if blue_epa is not None:
            out["blue_alliance_score"] = ScorePrediction(mean=blue_epa)

        if red_epa is not None and blue_epa is not None and win_prob is not None:
            if win_prob >= 0.5:
                out["winner"] = BinaryPrediction(predicted="red",  confidence=win_prob)
            else:
                out["winner"] = BinaryPrediction(predicted="blue", confidence=1.0 - win_prob)

        for color in ("red", "blue"):
            auto    = m.get(f"{color}_auto_epa_sum")
            teleop  = m.get(f"{color}_teleop_epa_sum")
            endgame = m.get(f"{color}_endgame_epa_sum")
            if auto    is not None: out[f"{color}_auto_score"]    = ScorePrediction(mean=auto)
            if teleop  is not None: out[f"{color}_teleop_score"]  = ScorePrediction(mean=teleop)
            if endgame is not None: out[f"{color}_endgame_score"] = ScorePrediction(mean=endgame)

        return out

    # ── Per-robot predictions from pre-event EPA ──────────────────────────────

    def _robot_predictions(self, team_key: str, te: dict) -> dict[str, Prediction]:
        # Only _start fields — pre-event baseline, no in-event leakage
        total   = te.get("epa_start") or 0.0
        auto    = te.get("auto_epa_start") or 0.0
        teleop  = te.get("teleop_epa_start") or 0.0
        endgame = te.get("endgame_epa_start") or 0.0

        if total == 0.0:
            return {}

        # Pre-event uncertainty is higher since we haven't seen in-event performance
        stdev = total * 0.15

        return {
            f"robot_total_contribution_{team_key}":   ScorePrediction(total,   stdev),
            f"robot_auto_contribution_{team_key}":    ScorePrediction(auto,    stdev * 0.4),
            f"robot_teleop_contribution_{team_key}":  ScorePrediction(teleop,  stdev * 0.7),
            f"robot_endgame_contribution_{team_key}": ScorePrediction(endgame, stdev * 0.3),
        }

    # ── Fetching with caching ─────────────────────────────────────────────────

    def _fetch_match(self, match_key: str) -> Optional[dict]:
        if match_key not in self._match_cache:
            try:
                self._match_cache[match_key] = self._client.get_match(match_key)
            except Exception as e:
                logger.debug("statbotics match miss %s: %s", match_key, e)
                self._match_cache[match_key] = None
        return self._match_cache[match_key]

    def _fetch_team_event(self, team_key: str, event_key: str) -> Optional[dict]:
        cache_key = (team_key, event_key)
        if cache_key not in self._team_event_cache:
            try:
                self._team_event_cache[cache_key] = self._client.get_team_event(
                    _strip_frc(team_key), event_key
                )
            except Exception as e:
                logger.debug("statbotics team_event miss %s @ %s: %s", team_key, event_key, e)
                self._team_event_cache[cache_key] = None
        return self._team_event_cache[cache_key]
