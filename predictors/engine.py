"""
predictors/engine.py — Core prediction logic.

Takes processed, tagged documents and computes probability-weighted
predictions for upcoming GST rule changes.

The model is intentionally transparent:
- Every prediction links to the signals that generated it
- Probabilities are derived from signal weights in config
- Backtest accuracy is tracked per signal type

How to add a new signal type:
1. Add it to SIGNAL_EVALUATORS below
2. Add its weight to config/sources.yaml signal_weights
3. Add backtest cases to tests/backtest_cases.json
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config" / "sources.yaml"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PREDICTIONS_DIR = Path(__file__).parent.parent / "data" / "predictions"


class Signal:
    """A single piece of evidence contributing to a prediction."""

    def __init__(
        self,
        signal_type: str,
        topic_id: str,
        strength: float,       # 0–1
        description: str,
        source_docs: list[str],
        horizon_days: int,
    ):
        self.signal_type = signal_type
        self.topic_id = topic_id
        self.strength = strength
        self.description = description
        self.source_docs = source_docs
        self.horizon_days = horizon_days


class Prediction:
    """A probability-weighted prediction for a topic changing."""

    def __init__(self, topic_id: str, topic_label: str):
        self.topic_id = topic_id
        self.topic_label = topic_label
        self.signals: list[Signal] = []
        self.probability: float = 0.0
        self.horizon_label: str = ""
        self.horizon_days: int = 0
        self.generated_at: str = datetime.utcnow().isoformat()

    def add_signal(self, signal: Signal):
        self.signals.append(signal)

    def compute_probability(self, weights: dict) -> float:
        """
        Combine signals into a final probability.
        Uses weighted average with diminishing returns on additional signals
        (the 3rd signal of the same type adds less than the 1st).
        """
        if not self.signals:
            return 0.0

        # Group by signal type for diminishing returns
        by_type = defaultdict(list)
        for s in self.signals:
            by_type[s.signal_type].append(s)

        total_weight = 0.0
        weighted_sum = 0.0

        for sig_type, sigs in by_type.items():
            base_weight = weights.get(sig_type, 0.10)
            for i, sig in enumerate(sigs):
                # Diminishing returns: 1st signal full weight, 2nd 60%, 3rd 40%
                decay = 1.0 / (1 + i * 0.6)
                w = base_weight * decay
                weighted_sum += sig.strength * w
                total_weight += w

        raw_prob = weighted_sum / total_weight if total_weight > 0 else 0
        # Cap at 95% — we never predict certainty
        self.probability = round(min(raw_prob * 100, 95), 1)
        return self.probability

    def compute_horizon(self) -> str:
        """Derive expected timeframe from signal types present."""
        sig_types = {s.signal_type for s in self.signals}

        if "council_deferred_item" in sig_types:
            self.horizon_days = 90
            self.horizon_label = "Next GST Council meeting"
        elif "budget_speech_phrase" in sig_types:
            self.horizon_days = 180
            self.horizon_label = "Next Union Budget / 2 Council meetings"
        elif "repeated_circular_topic" in sig_types:
            self.horizon_days = 180
            self.horizon_label = "2–3 quarters"
        else:
            self.horizon_days = 270
            self.horizon_label = "Next FY"

        return self.horizon_label

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "topic_label": self.topic_label,
            "probability": self.probability,
            "horizon_label": self.horizon_label,
            "horizon_days": self.horizon_days,
            "signal_count": len(self.signals),
            "signals": [
                {
                    "type": s.signal_type,
                    "strength": s.strength,
                    "description": s.description,
                    "source_docs": s.source_docs[:5],  # cap for readability
                }
                for s in self.signals
            ],
            "generated_at": self.generated_at,
        }


class PredictionEngine:
    """
    Main engine. Reads processed documents, evaluates signals, generates predictions.
    """

    def __init__(self):
        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)
        self.signal_weights = self.config["prediction"]["signal_weights"]
        self.min_prob = self.config["prediction"]["min_probability_to_surface"]
        self.min_signals = self.config["prediction"]["min_signals_to_predict"]
        self.topics = {t["id"]: t["label"] for t in self.config["topics"]}
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    def load_processed_docs(self) -> list[dict]:
        docs = []
        for path in PROCESSED_DIR.glob("*.json"):
            try:
                docs.append(json.loads(path.read_text()))
            except Exception:
                continue
        return docs

    def evaluate_repeated_circular_topic(
        self, docs: list[dict], topic_id: str
    ) -> Optional[Signal]:
        """
        Signal: same topic clarified 2+ times in CBIC circulars.
        Implies ongoing ambiguity — more clarification likely.
        """
        relevant = [
            d for d in docs
            if d.get("source_id") == "cbic_circulars"
            and topic_id in d.get("topic_tags", [])
        ]
        if len(relevant) < 2:
            return None

        # More circulars on same topic = stronger signal
        count = len(relevant)
        strength = min(0.3 + (count - 2) * 0.15, 0.9)

        return Signal(
            signal_type="repeated_circular_topic",
            topic_id=topic_id,
            strength=strength,
            description=f"{count} CBIC circulars on this topic — ongoing ambiguity suggests further clarification likely",
            source_docs=[d["doc_id"] for d in relevant[-5:]],
            horizon_days=180,
        )

    def evaluate_council_deferred_item(
        self, docs: list[dict], topic_id: str
    ) -> Optional[Signal]:
        """
        Signal: topic appears in GST Council minutes as deferred.
        Deferred items almost always resurface in next 1-2 meetings.
        """
        council_docs = [
            d for d in docs
            if d.get("source_id") == "gst_council_minutes"
            and topic_id in d.get("topic_tags", [])
        ]
        # Look for deferral language in content
        deferred = [
            d for d in council_docs
            if any(
                kw in (d.get("content") or "").lower()
                for kw in ["defer", "next meeting", "further deliberation", "examine further"]
            )
        ]
        if not deferred:
            return None

        return Signal(
            signal_type="council_deferred_item",
            topic_id=topic_id,
            strength=0.85,
            description=f"Item deferred in {len(deferred)} GST Council meeting(s) — high likelihood of resolution in next meeting",
            source_docs=[d["doc_id"] for d in deferred],
            horizon_days=90,
        )

    def evaluate_aar_ruling_frequency(
        self, docs: list[dict], topic_id: str
    ) -> Optional[Signal]:
        """
        Signal: 3+ AARs on same topic in rolling 12 months.
        High AAR frequency on a topic = disputed area = CBIC will clarify.
        """
        cutoff = datetime.utcnow() - timedelta(days=365)
        relevant = []
        for d in docs:
            if d.get("source_id") != "aar_rulings":
                continue
            if topic_id not in d.get("topic_tags", []):
                continue
            date_str = d.get("date")
            if date_str:
                try:
                    if datetime.fromisoformat(date_str) < cutoff:
                        continue
                except Exception:
                    pass
            relevant.append(d)

        if len(relevant) < 3:
            return None

        strength = min(0.2 + len(relevant) * 0.08, 0.75)
        return Signal(
            signal_type="aar_ruling_frequency",
            topic_id=topic_id,
            strength=strength,
            description=f"{len(relevant)} AAR rulings on this topic in last 12 months — judicial pressure likely to trigger CBIC clarification",
            source_docs=[d["doc_id"] for d in relevant[-5:]],
            horizon_days=180,
        )

    def evaluate_budget_speech_phrase(
        self, docs: list[dict], topic_id: str
    ) -> Optional[Signal]:
        """
        Signal: topic mentioned with action language in budget speech.
        Budget speech + topic mention reliably precedes council action.
        """
        budget_docs = [
            d for d in docs
            if d.get("source_id") == "budget_speeches"
            and topic_id in d.get("topic_tags", [])
        ]
        # Check for action phrases (not just mentions)
        action_phrases = [
            "rationalise", "simplify", "review", "examine", "bring within",
            "relief", "reduce", "exempt", "clarif"
        ]
        actionable = [
            d for d in budget_docs
            if any(p in (d.get("content") or "").lower() for p in action_phrases)
        ]
        if not actionable:
            return None

        # Most recent budget speech year
        years = [d.get("metadata", {}).get("year", "?") for d in actionable]
        return Signal(
            signal_type="budget_speech_phrase",
            topic_id=topic_id,
            strength=0.70,
            description=f"Budget speech ({', '.join(years)}) contained action language on this topic",
            source_docs=[d["doc_id"] for d in actionable],
            horizon_days=180,
        )

    def evaluate_industry_ask_repeat(
        self, docs: list[dict], topic_id: str
    ) -> Optional[Signal]:
        """
        Signal: same ask in 2+ consecutive industry memoranda.
        Persistent industry demand eventually gets addressed.
        """
        memo_docs = [
            d for d in docs
            if d.get("source_id") in ["icai_memoranda", "ficci_submissions"]
            and topic_id in d.get("topic_tags", [])
        ]
        if len(memo_docs) < 2:
            return None

        return Signal(
            signal_type="industry_ask_repeat",
            topic_id=topic_id,
            strength=0.45,
            description=f"Repeated ask across {len(memo_docs)} industry submissions — government typically addresses persistent asks within 2–3 budget cycles",
            source_docs=[d["doc_id"] for d in memo_docs],
            horizon_days=365,
        )

    def run(self) -> list[dict]:
        """Full prediction run. Returns list of prediction dicts."""
        docs = self.load_processed_docs()
        if not docs:
            print("[engine] no processed docs found — run ingest first")
            return []

        print(f"[engine] loaded {len(docs)} processed documents")

        evaluators = [
            self.evaluate_repeated_circular_topic,
            self.evaluate_council_deferred_item,
            self.evaluate_aar_ruling_frequency,
            self.evaluate_budget_speech_phrase,
            self.evaluate_industry_ask_repeat,
        ]

        predictions = []

        for topic_id, topic_label in self.topics.items():
            pred = Prediction(topic_id, topic_label)

            for evaluator in evaluators:
                signal = evaluator(docs, topic_id)
                if signal:
                    pred.add_signal(signal)

            if len(pred.signals) < self.min_signals:
                continue

            pred.compute_probability(self.signal_weights)
            pred.compute_horizon()

            if pred.probability >= self.min_prob:
                predictions.append(pred)

        # Sort by probability descending
        predictions.sort(key=lambda p: -p.probability)

        output = {
            "generated_at": datetime.utcnow().isoformat(),
            "doc_count": len(docs),
            "prediction_count": len(predictions),
            "predictions": [p.to_dict() for p in predictions],
        }

        # Save
        out_path = PREDICTIONS_DIR / "latest.json"
        out_path.write_text(json.dumps(output, indent=2))

        # Also save a timestamped snapshot
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
        snapshot_path = PREDICTIONS_DIR / f"snapshot_{ts}.json"
        snapshot_path.write_text(json.dumps(output, indent=2))

        print(f"[engine] generated {len(predictions)} predictions → {out_path}")
        return output["predictions"]
