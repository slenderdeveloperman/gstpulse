"""
predictors/backtest.py — Validates the model against known historical GST changes.

The backtest corpus is in tests/backtest_cases.json.
Each case is a known GST change with the signals that were present beforehand.

Running this proves the model's calibration and gives the README
a credibility claim: "X% of predictions in the 40-60% range materialised."

Add cases as you learn about new historical patterns.
"""

import json
from pathlib import Path
from predictors.engine import PredictionEngine, Prediction, Signal


BACKTEST_CASES_PATH = Path(__file__).parent.parent / "tests" / "backtest_cases.json"


def run_backtest() -> dict:
    """
    Runs all backtest cases and returns accuracy metrics by probability bucket.
    """
    cases = json.loads(BACKTEST_CASES_PATH.read_text())

    buckets = {
        "30-50": {"predicted": 0, "materialised": 0},
        "50-70": {"predicted": 0, "materialised": 0},
        "70-85": {"predicted": 0, "materialised": 0},
        "85+":   {"predicted": 0, "materialised": 0},
    }

    results = []

    for case in cases:
        # Reconstruct what the model would have predicted
        # given the signals that existed at the time
        pred = Prediction(case["topic_id"], case["topic_label"])
        for sig_data in case["signals_at_time"]:
            pred.add_signal(Signal(
                signal_type=sig_data["type"],
                topic_id=case["topic_id"],
                strength=sig_data["strength"],
                description=sig_data["description"],
                source_docs=sig_data.get("source_docs", []),
                horizon_days=sig_data.get("horizon_days", 180),
            ))

        engine = PredictionEngine()
        prob = pred.compute_probability(engine.signal_weights)
        horizon = pred.compute_horizon()
        materialised = case["materialised"]

        # Assign to bucket
        bucket = None
        if prob >= 85:
            bucket = "85+"
        elif prob >= 70:
            bucket = "70-85"
        elif prob >= 50:
            bucket = "50-70"
        elif prob >= 30:
            bucket = "30-50"

        if bucket:
            buckets[bucket]["predicted"] += 1
            if materialised:
                buckets[bucket]["materialised"] += 1

        results.append({
            "case": case["description"],
            "topic_id": case["topic_id"],
            "predicted_probability": prob,
            "predicted_horizon": horizon,
            "actual_materialised": materialised,
            "actual_lag_days": case.get("actual_lag_days"),
            "correct": (prob >= 50 and materialised) or (prob < 50 and not materialised),
        })

    # Compute accuracy per bucket
    accuracy = {}
    for bucket, counts in buckets.items():
        if counts["predicted"] > 0:
            accuracy[bucket] = {
                "n": counts["predicted"],
                "materialised": counts["materialised"],
                "accuracy": round(counts["materialised"] / counts["predicted"] * 100, 1),
            }

    overall_correct = sum(1 for r in results if r["correct"])
    overall_accuracy = round(overall_correct / len(results) * 100, 1) if results else 0

    return {
        "overall_accuracy": overall_accuracy,
        "case_count": len(results),
        "by_bucket": accuracy,
        "cases": results,
    }


if __name__ == "__main__":
    report = run_backtest()
    print(f"\n📊 Backtest Results ({report['case_count']} cases)")
    print(f"   Overall accuracy: {report['overall_accuracy']}%\n")
    for bucket, data in report["by_bucket"].items():
        print(f"   {bucket}% probability bucket: {data['accuracy']}% accuracy ({data['n']} cases)")
    print()
