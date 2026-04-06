from __future__ import annotations

import pytest
from core.types import IntentClass, RouterResult
from router.classifier import IntentRouter, _softmax

# ── _softmax ──────────────────────────────────────────────────────────────────


def test_softmax_sums_to_one() -> None:
    import numpy as np

    logits = np.array([1.0, 2.0, 0.5, -1.0])
    probs = _softmax(logits)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_softmax_argmax_preserved() -> None:
    import numpy as np

    logits = np.array([0.1, 5.0, 0.3, 0.2])
    probs = _softmax(logits)
    assert int(probs.argmax()) == 1


# ── keyword fallback ──────────────────────────────────────────────────────────


@pytest.fixture()
def router() -> IntentRouter:
    """Router with no ONNX model — uses keyword fallback."""
    from pathlib import Path

    from core.config import Settings

    s = Settings(router_model_path=Path("/nonexistent/router.onnx"))
    return IntentRouter(settings=s)


def test_classify_spiritual(router: IntentRouter) -> None:
    result = router.classify("What does the Bible say about prayer?")
    assert result.intent == IntentClass.SPIRITUAL
    assert result.confidence > 0


def test_classify_career(router: IntentRouter) -> None:
    result = router.classify("Find me ML Engineer jobs in Dallas")
    assert result.intent == IntentClass.CAREER


def test_classify_intelligence(router: IntentRouter) -> None:
    result = router.classify("What is NVDA stock price today?")
    assert result.intent == IntentClass.INTELLIGENCE


def test_classify_creative(router: IntentRouter) -> None:
    result = router.classify("Write a YouTube script about LangGraph")
    assert result.intent == IntentClass.CREATIVE


def test_classify_returns_router_result(router: IntentRouter) -> None:
    result = router.classify("Help me find a verse about strength")
    assert isinstance(result, RouterResult)
    assert 0.0 <= result.confidence <= 1.0


def test_classify_empty_raises(router: IntentRouter) -> None:
    from core.exceptions import RouterError

    with pytest.raises(RouterError):
        router.classify("")


def test_classify_whitespace_raises(router: IntentRouter) -> None:
    from core.exceptions import RouterError

    with pytest.raises(RouterError):
        router.classify("   ")


def test_classify_unknown_defaults_to_intelligence(router: IntentRouter) -> None:
    result = router.classify("xyzzy frobnicator quux")
    assert result.intent == IntentClass.INTELLIGENCE
    assert result.confidence < 0.7
