from __future__ import annotations

from core.types import IntentClass, RouterResult, SquadState


def test_intent_class_values() -> None:
    assert IntentClass.SPIRITUAL == "spiritual"
    assert IntentClass.CAREER == "career"
    assert IntentClass.INTELLIGENCE == "intelligence"
    assert IntentClass.CREATIVE == "creative"


def test_intent_class_is_str() -> None:
    assert isinstance(IntentClass.SPIRITUAL, str)


def test_router_result_confident() -> None:
    r = RouterResult(intent=IntentClass.SPIRITUAL, confidence=0.9)
    assert r.is_confident() is True
    assert r.is_confident(threshold=0.95) is False


def test_router_result_not_confident() -> None:
    r = RouterResult(intent=IntentClass.INTELLIGENCE, confidence=0.4)
    assert r.is_confident() is False


def test_router_result_immutable() -> None:
    r = RouterResult(intent=IntentClass.CAREER, confidence=0.8)
    with pytest.raises(Exception):
        r.intent = IntentClass.CREATIVE  # type: ignore[misc]


def test_squad_state_protocol_satisfied_by_dict() -> None:
    state: dict = {"intent": "career", "messages": []}
    assert isinstance(state, SquadState)


import pytest
