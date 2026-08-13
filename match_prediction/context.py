"""
MatchContext — the single object broadcast to every model.
Models read whatever they need and ignore the rest.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TeamStats:
    team: str
    matches_played: int
    avg_total_score: float
    avg_auto_score: float
    avg_teleop_score: float
    avg_endgame_score: float
    score_variance: float
    win_rate: float
    # Raw match history for models that want to do their own computation
    match_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MatchContext:
    # Match identity
    event_key: str
    match_key: str
    match_number: int
    comp_level: str   # "qm", "qf", "sf", "f"

    # Alliance compositions (team keys, e.g. "frc254")
    red_alliance: list[str]
    blue_alliance: list[str]

    # Pre-computed per-team stats (keyed by team key)
    team_stats: dict[str, TeamStats]

    # All completed match results at this event so far (for in-event calibration)
    event_match_results: list[dict[str, Any]] = field(default_factory=list)

    # Global/historical data blob — models can pull TBA stats, EPA tables, etc.
    # Structure is model-defined; ensemble just passes it through
    global_context: dict[str, Any] = field(default_factory=dict)

    # Number of qual matches played at this event (useful for cold-start detection)
    event_matches_played: int = 0

    def teams_for_alliance(self, color: str) -> list[str]:
        return self.red_alliance if color == "red" else self.blue_alliance

    def stats_for_alliance(self, color: str) -> list[TeamStats]:
        teams = self.teams_for_alliance(color)
        return [self.team_stats[t] for t in teams if t in self.team_stats]
