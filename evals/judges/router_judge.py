"""Router eval judge — tests keyword fallback accuracy against labeled examples.

Does NOT require an LLM or ONNX model — runs entirely on the keyword classifier.
Useful for CI validation that routing logic is correct before ONNX training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RouterResult:
    text: str
    expected: str
    predicted: str
    confidence: float
    passed: bool
    confidence_ok: bool


def evaluate(dataset_path: Path) -> list[RouterResult]:
    """Run router eval against a JSONL dataset."""
    import json

    from core.config import Settings
    from router.classifier import IntentRouter

    # Use keyword fallback (no ONNX required)
    settings = Settings(router_model_path=Path("/nonexistent/router.onnx"))
    router = IntentRouter(settings=settings)

    results: list[RouterResult] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row["text"]
            expected = row["expected_intent"]
            min_conf = row.get("expected_confidence_min", 0.5)

            result = router.classify(text)
            predicted = result.intent.value
            confidence = result.confidence

            results.append(
                RouterResult(
                    text=text,
                    expected=expected,
                    predicted=predicted,
                    confidence=confidence,
                    passed=(predicted == expected),
                    confidence_ok=(confidence >= min_conf),
                )
            )

    return results


def score(results: list[RouterResult]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.passed)
    conf_ok = sum(1 for r in results if r.confidence_ok)
    accuracy = correct / total if total > 0 else 0.0
    failures = [r for r in results if not r.passed]
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "confidence_met": conf_ok,
        "failures": [
            {
                "text": r.text[:60],
                "expected": r.expected,
                "got": r.predicted,
                "conf": round(r.confidence, 3),
            }
            for r in failures
        ],
    }
