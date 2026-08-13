"""
Registry of all possible output keys any model can claim to predict.
Each output declares its type (Binary or Score) and metadata.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OutputKind(Enum):
    BINARY = "binary"   # two-outcome prediction — carries predicted + confidence
    SCORE  = "score"    # numeric quantity — carries mean + stdev


@dataclass(frozen=True)
class OutputSpec:
    key: str
    kind: OutputKind
    description: str
    # For Binary outputs: the two legal outcome labels
    outcomes: tuple[str, str] | None = None
    # If True, key is a template — models suffix it with a team key, e.g. key_frc254
    per_robot: bool = False


# ── Master output registry ──────────────────────────────────────────────────

OUTPUT_REGISTRY: dict[str, OutputSpec] = {
    spec.key: spec for spec in [

        # ── Binary outputs ──────────────────────────────────────────────────

        OutputSpec(
            key="winner",
            kind=OutputKind.BINARY,
            outcomes=("red", "blue"),
            description="Which alliance wins the match",
        ),

        # ── Alliance score outputs ──────────────────────────────────────────

        OutputSpec(
            key="red_alliance_score",
            kind=OutputKind.SCORE,
            description="Total score for the red alliance",
        ),
        OutputSpec(
            key="blue_alliance_score",
            kind=OutputKind.SCORE,
            description="Total score for the blue alliance",
        ),
        OutputSpec(
            key="score_margin",
            kind=OutputKind.SCORE,
            description="Red alliance score minus blue alliance score",
        ),
        OutputSpec(
            key="red_auto_score",
            kind=OutputKind.SCORE,
            description="Red alliance auto period score",
        ),
        OutputSpec(
            key="blue_auto_score",
            kind=OutputKind.SCORE,
            description="Blue alliance auto period score",
        ),
        OutputSpec(
            key="red_teleop_score",
            kind=OutputKind.SCORE,
            description="Red alliance teleop period score",
        ),
        OutputSpec(
            key="blue_teleop_score",
            kind=OutputKind.SCORE,
            description="Blue alliance teleop period score",
        ),
        OutputSpec(
            key="red_endgame_score",
            kind=OutputKind.SCORE,
            description="Red alliance endgame score",
        ),
        OutputSpec(
            key="blue_endgame_score",
            kind=OutputKind.SCORE,
            description="Blue alliance endgame score",
        ),

        # ── Per-robot outputs (per_robot=True — append _{team} when producing) ──

        OutputSpec(
            key="robot_auto_contribution",
            kind=OutputKind.SCORE,
            description="Expected auto score contribution for a robot",
            per_robot=True,
        ),
        OutputSpec(
            key="robot_teleop_contribution",
            kind=OutputKind.SCORE,
            description="Expected teleop score contribution for a robot",
            per_robot=True,
        ),
        OutputSpec(
            key="robot_endgame_contribution",
            kind=OutputKind.SCORE,
            description="Expected endgame score contribution for a robot",
            per_robot=True,
        ),
        OutputSpec(
            key="robot_total_contribution",
            kind=OutputKind.SCORE,
            description="Expected total score contribution for a robot",
            per_robot=True,
        ),
    ]
}


def get_spec(key: str) -> Optional[OutputSpec]:
    """Return the OutputSpec for a key, resolving per_robot template keys."""
    if key in OUTPUT_REGISTRY:
        return OUTPUT_REGISTRY[key]
    for spec in OUTPUT_REGISTRY.values():
        if spec.per_robot and key.startswith(spec.key + "_"):
            return spec
    return None


def validate_output_key(key: str) -> bool:
    return get_spec(key) is not None
