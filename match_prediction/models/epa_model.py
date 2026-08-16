"""
EPAModel — Statbotics-style Expected Points Added, scoped to this event only.

Ported from avgupta456/statbotics (backend/src/models/epa/{math,constants}.py),
MIT License, Copyright (c) 2020 Abhijit Gupta:
https://github.com/avgupta456/statbotics/blob/master/LICENSE
The permission notice above is reproduced here as required by that license.

The rating loop is unchanged from the original: each team's EPA is an
exponentially-weighted moving average of its point contribution, split into
total/auto/teleop/endgame. After every match, each team's rating is nudged
toward its actual contribution that match — early matches move the rating a
lot (`percent_func` starts high and decays over a team's first 6 matches),
elimination matches count for 1/3 weight since they're noisier per-match, and
match win probability comes from a logistic function of the predicted score
margin scaled by the event's own score standard deviation.

What's deliberately NOT here, by design (not just left out):

- No cross-season priors. Statbotics blends a team's last two years' EPA and
  mean-reverts toward a rookie baseline (init.py). We don't — cross-year and
  cross-event carryover is StatboticsModel's job (it reads Statbotics' own
  precomputed epa_start, which already has full history behind it). This
  model only ever sees context.event_match_results, i.e. this event.
- No synthetic seed for a team we haven't observed yet. A team's rating
  simply doesn't exist until they've played >= 1 match at this event; it's
  seeded directly from that match's actual result, not a guessed average.
  Any output that needs a team with no rating yet is omitted rather than
  backfilled with a fabricated number — early in an event this model will
  often return {} or a partial dict, and the ensemble's per-key fusion
  (raw[model_id] only contributes keys it actually produced) already handles
  models sitting out cleanly. Other models (StatboticsModel, TBA-average-
  style models) that don't need in-event history fill that gap instead, and
  WeightTracker defaults to equal weighting until enough samples exist to
  judge accuracy — nothing here needs to hand-hold that.

Also not ported: the year-specific score-reconstruction quirks in Statbotics'
breakdown.py (2018 switch/scale sigmoids, 2023 grid overflow, 2025 processor
algae double-counting) — those encode each season's exact TBA
score_breakdown schema by hand. This model works off the generic
{total, auto, teleop, endgame} shape instead. Also not ported: cross-season
unitless normalization (unitless.py) — not needed for single-event predictions.
"""

from collections import defaultdict

import numpy as np

from match_prediction.base_model import BaseModel
from match_prediction.context import MatchContext
from match_prediction.predictions import BinaryPrediction, Prediction, ScorePrediction

# ── Constants (statbotics/backend/src/models/epa/constants.py) ───────────────

ELIM_WEIGHT = 1 / 3    # elim matches count for this fraction of a normal rating update
K_FACTOR = -5 / 8      # win-probability logistic steepness (statbotics' post-2008 k_func)
MIN_MATCHES_FOR_SD = 4 # need at least this many completed matches to trust an event score_sd

# Component vector layout, matching statbotics' [epa, auto_epa, teleop_epa, endgame_epa]
TOTAL, AUTO, TELEOP, ENDGAME = 0, 1, 2, 3


# ── Core rating (statbotics/backend/src/models/epa/math.py) ──────────────────

class EPARating:
    """
    Exponentially-weighted moving average + matching variance, over the 4-vector
    [total, auto, teleop, endgame]. The variance EWMA is a natural extension of
    statbotics' own model (their math.py docstring already frames the mean update
    as the posterior of a moving-average process) — we track it so per-robot
    predictions carry a real uncertainty instead of a hand-picked constant.
    """

    __slots__ = ("mean", "var")

    def __init__(self, mean: np.ndarray, var: np.ndarray):
        self.mean = mean
        self.var = var

    def add_obs(self, x: np.ndarray, percent: float, weight: float) -> None:
        delta = x - self.mean
        new_mean = (1 - percent) * self.mean + percent * x
        new_var = (1 - percent) * self.var + percent * delta ** 2
        self.mean = weight * new_mean + (1 - weight) * self.mean
        self.var = weight * new_var + (1 - weight) * self.var


def percent_func(match_count: int) -> float:
    """Learning rate: starts at 1/3, decays to 1/5 over a team's first 6 matches."""
    prev = min(0.5, max(0.3, 0.5 - 0.2 / 6 * (match_count - 6)))
    return 2 / 3 * prev


# ── Model ──────────────────────────────────────────────────────────────────

class EPAModel(BaseModel):
    claimed_outputs = [
        "winner",
        "red_alliance_score",
        "blue_alliance_score",
        "red_auto_score",
        "blue_auto_score",
        "red_teleop_score",
        "blue_teleop_score",
        "red_endgame_score",
        "blue_endgame_score",
        "robot_auto_contribution",
        "robot_teleop_contribution",
        "robot_endgame_contribution",
        "robot_total_contribution",
    ]

    def can_predict(self, context: MatchContext) -> bool:
        return True

    def predict(self, context: MatchContext) -> dict[str, Prediction]:
        ratings = self._replay_event(context)
        result: dict[str, Prediction] = {}

        for team in context.red_alliance + context.blue_alliance:
            r = ratings.get(team)
            if r is None:
                continue
            result[f"robot_total_contribution_{team}"] = ScorePrediction(r.mean[TOTAL], np.sqrt(r.var[TOTAL]))
            result[f"robot_auto_contribution_{team}"] = ScorePrediction(r.mean[AUTO], np.sqrt(r.var[AUTO]))
            result[f"robot_teleop_contribution_{team}"] = ScorePrediction(r.mean[TELEOP], np.sqrt(r.var[TELEOP]))
            result[f"robot_endgame_contribution_{team}"] = ScorePrediction(r.mean[ENDGAME], np.sqrt(r.var[ENDGAME]))

        red_teams = [t for t in context.red_alliance if t in ratings]
        blue_teams = [t for t in context.blue_alliance if t in ratings]
        if len(red_teams) < len(context.red_alliance) or len(blue_teams) < len(context.blue_alliance):
            # At least one robot in this match has no in-event rating yet — an
            # alliance sum would silently drop them, so don't produce alliance
            # or match-level outputs at all; let other models cover this match.
            return result

        red_mean = sum(ratings[t].mean for t in red_teams)
        blue_mean = sum(ratings[t].mean for t in blue_teams)
        red_var = sum(ratings[t].var for t in red_teams)
        blue_var = sum(ratings[t].var for t in blue_teams)

        for color, mean, var in [("red", red_mean, red_var), ("blue", blue_mean, blue_var)]:
            result[f"{color}_alliance_score"] = ScorePrediction(mean[TOTAL], np.sqrt(var[TOTAL]))
            result[f"{color}_auto_score"] = ScorePrediction(mean[AUTO], np.sqrt(var[AUTO]))
            result[f"{color}_teleop_score"] = ScorePrediction(mean[TELEOP], np.sqrt(var[TELEOP]))
            result[f"{color}_endgame_score"] = ScorePrediction(mean[ENDGAME], np.sqrt(var[ENDGAME]))

        score_sd = self._estimate_score_sd(context)
        if score_sd is not None and score_sd > 0:
            norm_diff = (red_mean[TOTAL] - blue_mean[TOTAL]) / score_sd
            red_win_prob = 1 / (1 + 10 ** (K_FACTOR * norm_diff))
            if red_win_prob >= 0.5:
                result["winner"] = BinaryPrediction(predicted="red", confidence=red_win_prob)
            else:
                result["winner"] = BinaryPrediction(predicted="blue", confidence=1.0 - red_win_prob)
        # else: too few matches at this event to trust a score_sd estimate —
        # omit winner rather than guess one; scores above are still reported.

        return result

    # ── Online update, replayed over the event's completed matches ──────────

    def _replay_event(self, context: MatchContext) -> dict[str, EPARating]:
        ratings: dict[str, EPARating] = {}
        counts: dict[str, int] = defaultdict(int)

        matches = sorted(context.event_match_results, key=lambda m: m.get("match_number", 0))
        for m in matches:
            red, blue = m.get("red", []), m.get("blue", [])
            red_bd, blue_bd = m.get("red_breakdown"), m.get("blue_breakdown")
            if not red or not blue or red_bd is None or blue_bd is None:
                continue
            elim = m.get("comp_level", "qm") != "qm"
            self._update_alliance(ratings, counts, red, self._breakdown_vector(red_bd), elim)
            self._update_alliance(ratings, counts, blue, self._breakdown_vector(blue_bd), elim)

        return ratings

    @staticmethod
    def _update_alliance(
        ratings: dict[str, EPARating],
        counts: dict[str, int],
        teams: list[str],
        actual: np.ndarray,
        elim: bool,
    ) -> None:
        # Teams new to this event start from a zero rating: their first
        # attributed value (below) becomes their observed EPA directly, not
        # a blend against a guessed prior.
        for t in teams:
            if t not in ratings:
                ratings[t] = EPARating(np.zeros(4), np.zeros(4))

        predicted = sum(ratings[t].mean for t in teams)
        err = (actual - predicted) / len(teams)
        weight = ELIM_WEIGHT if elim else 1.0
        for t in teams:
            attrib = ratings[t].mean + err
            if counts[t] == 0:
                # First-ever observation for this team: take it at face value
                # rather than blending against the meaningless zero placeholder
                # mean (which would otherwise register as a huge fake variance
                # via delta**2, and — for an elim-weighted first match — as an
                # artificially damped mean).
                ratings[t].mean = attrib
                ratings[t].var = np.zeros(4)
            else:
                ratings[t].add_obs(attrib, percent_func(counts[t]), weight)
            if not elim:
                counts[t] += 1

    @staticmethod
    def _breakdown_vector(bd: dict[str, float]) -> np.ndarray:
        auto = bd.get("auto", 0.0)
        teleop = bd.get("teleop", 0.0)
        endgame = bd.get("endgame", 0.0)
        total = bd.get("total", auto + teleop + endgame)
        return np.array([total, auto, teleop, endgame], dtype=float)

    @staticmethod
    def _estimate_score_sd(context: MatchContext) -> float | None:
        totals = []
        for m in context.event_match_results:
            red_bd, blue_bd = m.get("red_breakdown"), m.get("blue_breakdown")
            if red_bd is not None:
                totals.append(EPAModel._breakdown_vector(red_bd)[TOTAL])
            if blue_bd is not None:
                totals.append(EPAModel._breakdown_vector(blue_bd)[TOTAL])
        if len(totals) < MIN_MATCHES_FOR_SD:
            return None
        return float(np.std(totals))
