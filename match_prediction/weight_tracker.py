"""
WeightTracker — per-model per-output accuracy tracking.

Scoring:
  Binary → Brier score: (predicted_prob - actual)² where actual ∈ {0, 1}
  Score  → MSE on the mean: (predicted_mean - actual_mean)²
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from match_prediction.outputs import OutputKind, get_spec
from match_prediction.predictions import BinaryPrediction, Prediction, ScorePrediction


@dataclass
class _ErrorWindow:
    errors: deque = field(default_factory=lambda: deque(maxlen=30))

    def push(self, error: float):
        self.errors.append(error)

    def mean_error(self) -> Optional[float]:
        if not self.errors:
            return None
        return sum(self.errors) / len(self.errors)

    def sample_count(self) -> int:
        return len(self.errors)


DEFAULT_WEIGHT = 0.5
MIN_SAMPLES_FOR_WEIGHT = 3


class WeightTracker:
    def __init__(self):
        self._windows: dict[str, dict[str, _ErrorWindow]] = defaultdict(
            lambda: defaultdict(_ErrorWindow)
        )

    def record_result(
        self,
        model_id: str,
        output_key: str,
        predicted: Prediction,
        actual: Prediction,
    ):
        spec = get_spec(output_key)
        if spec is None:
            return

        if spec.kind == OutputKind.BINARY:
            if not isinstance(predicted, BinaryPrediction) or not isinstance(actual, BinaryPrediction):
                return
            outcome_a = spec.outcomes[0]
            pred_prob = predicted.probability_of(outcome_a)
            actual_prob = actual.probability_of(outcome_a)  # 0.0 or 1.0
            error = (pred_prob - actual_prob) ** 2  # Brier score

        else:  # SCORE
            if not isinstance(predicted, ScorePrediction) or not isinstance(actual, ScorePrediction):
                return
            error = (predicted.mean - actual.mean) ** 2  # MSE on mean

        self._windows[model_id][output_key].push(error)

    def get_weight(self, model_id: str, output_key: str) -> float:
        window = self._windows[model_id][output_key]
        if window.sample_count() < MIN_SAMPLES_FOR_WEIGHT:
            return DEFAULT_WEIGHT
        mean_err = window.mean_error()
        if mean_err is None or mean_err == 0:
            return 1.0
        return 1.0 / (mean_err + 1e-6)

    def get_weights_for_output(
        self, model_ids: list[str], output_key: str
    ) -> dict[str, float]:
        raw = {mid: self.get_weight(mid, output_key) for mid in model_ids}
        total = sum(raw.values())
        if total == 0:
            uniform = 1.0 / len(model_ids) if model_ids else 0.0
            return {mid: uniform for mid in model_ids}
        return {mid: w / total for mid, w in raw.items()}

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            model_id: {key: self.get_weight(model_id, key) for key in outputs}
            for model_id, outputs in self._windows.items()
        }
