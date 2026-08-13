"""
Prediction value types. Every model output is one of these, not a raw float.

Binary  — a two-outcome prediction (win/loss, completes/fails)
           carries: which option is predicted + confidence [0, 1]

Score   — any numeric game quantity (points, count, rate)
           carries: mean (expected value) + stdev (uncertainty)
"""

from dataclasses import dataclass


@dataclass
class BinaryPrediction:
    """
    A prediction between exactly two labeled outcomes.
    `predicted` is the name of the favored outcome.
    `confidence` is the probability assigned to that outcome, in [0.5, 1.0].
    """
    predicted: str          # e.g. "red", "blue", "yes", "no"
    confidence: float       # probability of `predicted` being correct

    def probability_of(self, outcome: str) -> float:
        if outcome == self.predicted:
            return self.confidence
        return 1.0 - self.confidence


@dataclass
class ScorePrediction:
    """
    A prediction of a numeric score/quantity.
    `mean` is the expected value, `stdev` is the uncertainty (1-sigma).
    """
    mean: float
    stdev: float = 0.0

    def variance(self) -> float:
        return self.stdev ** 2


# Union type for all prediction values
Prediction = BinaryPrediction | ScorePrediction
