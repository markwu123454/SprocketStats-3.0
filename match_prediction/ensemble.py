"""
Ensemble — broadcasts MatchContext to all registered models,
collects typed partial outputs, and fuses them per output kind.

Binary fusion:  confidence-weighted probability average → BinaryPrediction
Score fusion:   weighted mixture of Gaussians → ScorePrediction (mean + stdev)
"""

import math
from dataclasses import dataclass, field

from match_prediction.base_model import BaseModel
from match_prediction.context import MatchContext
from match_prediction.outputs import OutputKind, get_spec
from match_prediction.predictions import BinaryPrediction, Prediction, ScorePrediction
from match_prediction.weight_tracker import WeightTracker


@dataclass
class PredictionResult:
    values: dict[str, Prediction]
    contributors: dict[str, list[str]]
    raw: dict[str, dict[str, Prediction]] = field(default_factory=dict)


class Ensemble:
    def __init__(self):
        self._models: list[BaseModel] = []
        self.weight_tracker = WeightTracker()

    def register(self, model: BaseModel):
        self._models.append(model)

    def predict(self, context: MatchContext) -> PredictionResult:
        # 1. Broadcast to all eligible models
        raw: dict[str, dict[str, Prediction]] = {}
        for model in self._models:
            if not model.can_predict(context):
                continue
            try:
                raw[model.model_id] = model.predict(context)
            except Exception as e:
                print(f"[Ensemble] {model.model_id} raised {e!r}, skipping")

        # 2. Collect all output keys across all model outputs
        all_keys: set[str] = set()
        for outputs in raw.values():
            all_keys.update(outputs.keys())

        # 3. Fuse per key
        fused: dict[str, Prediction] = {}
        contributors: dict[str, list[str]] = {}

        for key in all_keys:
            spec = get_spec(key)
            if spec is None:
                continue

            producing = [mid for mid, outputs in raw.items() if key in outputs]
            if not producing:
                continue

            weights = self.weight_tracker.get_weights_for_output(producing, key)

            if spec.kind == OutputKind.BINARY:
                fused[key] = _fuse_binary(key, producing, raw, weights, spec)
            else:
                fused[key] = _fuse_score(key, producing, raw, weights)

            contributors[key] = producing

        return PredictionResult(values=fused, contributors=contributors, raw=raw)

    def record_actuals(
        self,
        prediction: PredictionResult,
        actuals: dict[str, Prediction],
    ):
        """
        Call after a match resolves with ground-truth values.
        For Binary: BinaryPrediction(predicted=actual_winner, confidence=1.0)
        For Score:  ScorePrediction(mean=actual_score, stdev=0)
        """
        for key, actual in actuals.items():
            for model_id in prediction.contributors.get(key, []):
                pred = prediction.raw.get(model_id, {}).get(key)
                if pred is not None:
                    self.weight_tracker.record_result(model_id, key, pred, actual)

    @property
    def models(self) -> list[BaseModel]:
        return list(self._models)


# ── Fusion ───────────────────────────────────────────────────────────────────

def _fuse_binary(
    key: str,
    producing: list[str],
    raw: dict[str, dict[str, Prediction]],
    weights: dict[str, float],
    spec,
) -> BinaryPrediction:
    """
    Weighted average of per-outcome probabilities.
    Weights are normalized so prob_a stays in [0, 1].
    """
    outcome_a, outcome_b = spec.outcomes
    prob_a = 0.0
    for mid in producing:
        pred = raw[mid][key]
        if isinstance(pred, BinaryPrediction):
            prob_a += pred.probability_of(outcome_a) * weights[mid]

    if prob_a >= 0.5:
        return BinaryPrediction(predicted=outcome_a, confidence=prob_a)
    return BinaryPrediction(predicted=outcome_b, confidence=1.0 - prob_a)


def _fuse_score(
    key: str,
    producing: list[str],
    raw: dict[str, dict[str, Prediction]],
    weights: dict[str, float],
) -> ScorePrediction:
    """
    Mixture of Gaussians:
      fused_mean = Σ w_i * mean_i
      fused_var  = Σ w_i * (var_i + mean_i²) − fused_mean²
    Preserves both the central estimate and the spread across disagreeing models.
    """
    items: list[tuple[float, float, float]] = []  # (mean, variance, weight)
    for mid in producing:
        pred = raw[mid][key]
        if isinstance(pred, ScorePrediction):
            items.append((pred.mean, pred.variance(), weights[mid]))

    if not items:
        return ScorePrediction(mean=0.0, stdev=0.0)

    fused_mean = sum(m * w for m, _, w in items)
    fused_var = sum((v + m ** 2) * w for m, v, w in items) - fused_mean ** 2
    return ScorePrediction(mean=fused_mean, stdev=math.sqrt(max(fused_var, 0.0)))
