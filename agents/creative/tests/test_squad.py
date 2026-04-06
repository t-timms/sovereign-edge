from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from creative.squad import CreativeSquad
from langchain_core.messages import HumanMessage


def _make_state(query: str, memory_ctx: str = "") -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "memory_context": memory_ctx,
        "intent": "creative",
        "intent_confidence": 0.85,
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }


# ── routing ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_run_routes_to_script() -> None:
    squad = CreativeSquad.__new__(CreativeSquad)
    squad._llm = MagicMock()
    squad._llm.complete = AsyncMock(return_value="Script outline here")

    result = await squad.run(_make_state("Write a YouTube script about LangGraph"))  # type: ignore[arg-type]

    assert result == "Script outline here"
    squad._llm.complete.assert_called_once()


@pytest.mark.asyncio()
async def test_run_routes_to_diagram() -> None:
    squad = CreativeSquad.__new__(CreativeSquad)
    squad._llm = MagicMock()
    squad._llm.complete = AsyncMock(return_value="d2\ndirection: right\n...")

    result = await squad.run(_make_state("Create a diagram for the LangGraph architecture"))  # type: ignore[arg-type]

    assert "d2" in result


@pytest.mark.asyncio()
async def test_run_routes_to_social() -> None:
    squad = CreativeSquad.__new__(CreativeSquad)
    squad._llm = MagicMock()
    squad._llm.complete = AsyncMock(return_value="LinkedIn post draft")

    result = await squad.run(_make_state("Draft a LinkedIn post about my LangGraph project"))  # type: ignore[arg-type]

    assert result == "LinkedIn post draft"


@pytest.mark.asyncio()
async def test_run_routes_to_general_creative() -> None:
    squad = CreativeSquad.__new__(CreativeSquad)
    squad._llm = MagicMock()
    squad._llm.complete = AsyncMock(return_value="Creative response")

    result = await squad.run(_make_state("Help me brainstorm ideas for a talk"))  # type: ignore[arg-type]

    assert result == "Creative response"


@pytest.mark.asyncio()
async def test_run_empty_messages_does_not_raise() -> None:
    squad = CreativeSquad.__new__(CreativeSquad)
    squad._llm = MagicMock()
    squad._llm.complete = AsyncMock(return_value="response")

    state = {
        "messages": [],
        "memory_context": "",
        "intent": "creative",
        "intent_confidence": 0.5,
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    result = await squad.run(state)  # type: ignore[arg-type]
    assert isinstance(result, str)


# ── memory_ctx injected ───────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_run_injects_memory_context() -> None:
    """Memory context string must reach the LLM prompt."""
    squad = CreativeSquad.__new__(CreativeSquad)
    squad._llm = MagicMock()
    captured_messages: list = []

    async def capture(messages: list, **kwargs: object) -> str:
        captured_messages.extend(messages)
        return "ok"

    squad._llm.complete = capture

    await squad.run(_make_state("Write a video script", memory_ctx="John likes direct style"))  # type: ignore[arg-type]

    assert any("John likes direct style" in str(m.content) for m in captured_messages)
