"""
BaseModel — the interface every prediction model must implement.
"""

from abc import ABC, abstractmethod

from match_prediction.context import MatchContext
from match_prediction.outputs import OUTPUT_REGISTRY, validate_output_key


class BaseModel(ABC):
    # Each subclass declares what outputs it produces
    # Use bare keys for alliance/match-level outputs: "red_win_probability"
    # Use template keys for per-robot outputs: "robot_teleop_contribution"
    # (the ensemble will look for per-robot keys with team suffixes)
    claimed_outputs: list[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for key in cls.claimed_outputs:
            if not validate_output_key(key):
                raise ValueError(
                    f"{cls.__name__} claimed unknown output '{key}'. "
                    f"Register it in outputs.py first."
                )

    @property
    def model_id(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def predict(self, context: MatchContext) -> dict[str, float]:
        """
        Return a dict of output_key -> predicted_value.
        Only return keys that are in claimed_outputs (or per-robot expansions of them).
        Missing keys are fine — the ensemble handles absent predictions gracefully.
        """
        ...

    def can_predict(self, context: MatchContext) -> bool:
        """
        Override to signal when this model doesn't have enough data to be reliable.
        The ensemble will skip this model's outputs (not weight them as bad).
        """
        return True
