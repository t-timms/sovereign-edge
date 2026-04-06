from __future__ import annotations

import pytest

# ── _route_intent ─────────────────────────────────────────────────────────────


def test_route_intent_spiritual() -> None:
    from orchestrator.graph import _route_intent

    state = {
        "intent": "spiritual",
        "messages": [],
        "intent_confidence": 0.9,
        "memory_context": "",
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    assert _route_intent(state) == "spiritual"  # type: ignore[arg-type]


def test_route_intent_career() -> None:
    from orchestrator.graph import _route_intent

    state = {
        "intent": "career",
        "messages": [],
        "intent_confidence": 0.8,
        "memory_context": "",
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    assert _route_intent(state) == "career"  # type: ignore[arg-type]


def test_route_intent_intelligence() -> None:
    from orchestrator.graph import _route_intent

    state = {
        "intent": "intelligence",
        "messages": [],
        "intent_confidence": 0.7,
        "memory_context": "",
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    assert _route_intent(state) == "intelligence"  # type: ignore[arg-type]


def test_route_intent_creative() -> None:
    from orchestrator.graph import _route_intent

    state = {
        "intent": "creative",
        "messages": [],
        "intent_confidence": 0.85,
        "memory_context": "",
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    assert _route_intent(state) == "creative"  # type: ignore[arg-type]


def test_route_intent_unknown_defaults_to_intelligence() -> None:
    from orchestrator.graph import _route_intent

    state = {
        "intent": "unknown_garbage",
        "messages": [],
        "intent_confidence": 0.1,
        "memory_context": "",
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    assert _route_intent(state) == "intelligence"  # type: ignore[arg-type]


# ── _route_hitl ───────────────────────────────────────────────────────────────


def test_route_hitl_required_goes_to_hitl() -> None:
    from orchestrator.graph import _route_hitl

    state = {
        "intent": "career",
        "messages": [],
        "intent_confidence": 0.8,
        "memory_context": "",
        "squad_result": "result",
        "hitl_required": True,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    assert _route_hitl(state) == "hitl"  # type: ignore[arg-type]


def test_route_hitl_not_required_goes_to_delivery() -> None:
    from orchestrator.graph import _route_hitl

    state = {
        "intent": "spiritual",
        "messages": [],
        "intent_confidence": 0.9,
        "memory_context": "",
        "squad_result": "result",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    assert _route_hitl(state) == "delivery"  # type: ignore[arg-type]


# ── delivery_node ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_delivery_node_returns_result_when_approved() -> None:
    from orchestrator.graph import delivery_node

    state = {
        "intent": "intelligence",
        "messages": [],
        "intent_confidence": 0.7,
        "memory_context": "",
        "squad_result": "NVDA is at $900",
        "hitl_required": False,
        "hitl_approved": True,
        "schedule_trigger": None,
    }
    result = await delivery_node(state)  # type: ignore[arg-type]

    messages = result["messages"]
    assert len(messages) == 1
    assert messages[0].content == "NVDA is at $900"


@pytest.mark.asyncio()
async def test_delivery_node_returns_cancelled_when_rejected() -> None:
    from orchestrator.graph import delivery_node

    state = {
        "intent": "career",
        "messages": [],
        "intent_confidence": 0.8,
        "memory_context": "",
        "squad_result": "Apply to job at Google",
        "hitl_required": True,
        "hitl_approved": False,
        "schedule_trigger": None,
    }
    result = await delivery_node(state)  # type: ignore[arg-type]

    messages = result["messages"]
    assert len(messages) == 1
    assert messages[0].content == "Action cancelled."


@pytest.mark.asyncio()
async def test_delivery_node_passes_through_when_hitl_not_required() -> None:
    from orchestrator.graph import delivery_node

    state = {
        "intent": "spiritual",
        "messages": [],
        "intent_confidence": 0.9,
        "memory_context": "",
        "squad_result": "Bible verse here",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    result = await delivery_node(state)  # type: ignore[arg-type]

    # hitl_approved=None + hitl_required=False → result passes through (not False)
    messages = result["messages"]
    assert messages[0].content == "Bible verse here"
