from match_prediction import Ensemble, MatchContext, TeamStats
from match_prediction.models import OPRModel, EPAModel, WinRateModel, ComponentOPRModel
from match_prediction.predictions import BinaryPrediction, ScorePrediction


def build_ensemble() -> Ensemble:
    ensemble = Ensemble()
    ensemble.register(OPRModel())
    ensemble.register(EPAModel())
    ensemble.register(WinRateModel())
    ensemble.register(ComponentOPRModel())
    return ensemble


def make_context() -> MatchContext:
    teams = {
        "frc254":  TeamStats("frc254",  8, 72.0, 18.0, 44.0, 10.0, 25.0, 0.75, []),
        "frc1678": TeamStats("frc1678", 8, 68.0, 15.0, 42.0, 11.0, 30.0, 0.70, []),
        "frc971":  TeamStats("frc971",  8, 65.0, 14.0, 40.0, 11.0, 20.0, 0.65, []),
        "frc148":  TeamStats("frc148",  8, 60.0, 12.0, 38.0, 10.0, 35.0, 0.60, []),
        "frc118":  TeamStats("frc118",  8, 58.0, 11.0, 36.0, 11.0, 28.0, 0.55, []),
        "frc2056": TeamStats("frc2056", 8, 55.0, 10.0, 34.0, 11.0, 22.0, 0.50, []),
    }
    return MatchContext(
        event_key="2026casj",
        match_key="2026casj_qm42",
        match_number=42,
        comp_level="qm",
        red_alliance=["frc254", "frc1678", "frc971"],
        blue_alliance=["frc148", "frc118", "frc2056"],
        team_stats=teams,
        global_context={"opr": {t: s.avg_total_score for t, s in teams.items()}},
    )


if __name__ == "__main__":
    ensemble = build_ensemble()
    context = make_context()
    result = ensemble.predict(context)

    print("=== Fused Predictions ===")
    for key, pred in sorted(result.values.items()):
        contributors = ", ".join(result.contributors.get(key, []))
        if isinstance(pred, BinaryPrediction):
            print(f"  {key}: {pred.predicted} @ {pred.confidence:.1%}  [{contributors}]")
        elif isinstance(pred, ScorePrediction):
            print(f"  {key}: {pred.mean:.1f} ± {pred.stdev:.1f}  [{contributors}]")

    # Simulate match resolving: red wins 215-185
    actuals = {
        "winner":             BinaryPrediction(predicted="red", confidence=1.0),
        "red_alliance_score": ScorePrediction(mean=215.0),
        "blue_alliance_score": ScorePrediction(mean=185.0),
        "score_margin":       ScorePrediction(mean=30.0),
    }
    ensemble.record_actuals(result, actuals)
    print("\n=== Weights after 1 match ===")
    for model_id, outputs in ensemble.weight_tracker.summary().items():
        for key, w in outputs.items():
            print(f"  {model_id} / {key}: {w:.4f}")
