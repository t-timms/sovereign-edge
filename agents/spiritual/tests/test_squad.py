from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from spiritual.squad import SpiritualSquad


def _make_state(query: str, memory_ctx: str = "") -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "memory_context": memory_ctx,
        "intent": "spiritual",
        "intent_confidence": 0.9,
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }


# ── _format_rag_results ───────────────────────────────────────────────────────


def test_format_rag_results_empty() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    result = squad._format_rag_results([])
    assert "(No matching scripture found" in result


def test_format_rag_results_formats_verses() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    results = [
        {"ref": "John.3.16", "text": "For God so loved the world..."},
        {"ref": "Ps.23.1", "text": "The Lord is my shepherd..."},
    ]
    formatted = squad._format_rag_results(results)
    assert "John.3.16" in formatted
    assert "For God so loved the world" in formatted
    assert "Ps.23.1" in formatted


def test_format_rag_results_handles_missing_keys() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    results = [{"ref": "", "text": ""}]
    formatted = squad._format_rag_results(results)
    # Should not raise; returns a line with empty brackets
    assert formatted is not None


# ── routing ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_run_routes_to_devotional() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    squad._llm = MagicMock()
    squad._rag = MagicMock()
    squad._rag.search.return_value = []
    squad._llm.complete = AsyncMock(return_value="Morning devotional text")

    result = await squad.run(_make_state("Give me a morning devotional"))  # type: ignore[arg-type]

    assert result == "Morning devotional text"


@pytest.mark.asyncio()
async def test_run_routes_to_verse_lookup() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    squad._llm = MagicMock()
    squad._rag = MagicMock()
    squad._rag.lookup_verse.return_value = "For God so loved the world..."
    squad._llm.complete = AsyncMock(return_value="Here is the verse: John 3:16")

    result = await squad.run(_make_state("Look up John 3:16"))  # type: ignore[arg-type]

    assert "John 3:16" in result


@pytest.mark.asyncio()
async def test_run_routes_to_qa() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    squad._llm = MagicMock()
    squad._rag = MagicMock()
    squad._rag.search.return_value = [{"ref": "Ps.23.1", "text": "The Lord is my shepherd"}]
    squad._llm.complete = AsyncMock(return_value="David wrote Psalm 23 as a song of trust.")

    result = await squad.run(_make_state("Who wrote Psalm 23?"))  # type: ignore[arg-type]

    assert result == "David wrote Psalm 23 as a song of trust."


@pytest.mark.asyncio()
async def test_run_empty_messages_does_not_raise() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    squad._llm = MagicMock()
    squad._rag = MagicMock()
    squad._rag.search.return_value = []
    squad._llm.complete = AsyncMock(return_value="response")

    state = {
        "messages": [],
        "memory_context": "",
        "intent": "spiritual",
        "intent_confidence": 0.5,
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": None,
    }
    result = await squad.run(state)  # type: ignore[arg-type]
    assert isinstance(result, str)


# ── morning_verse ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_morning_verse_returns_formatted_string() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    squad._rag = MagicMock()
    squad._rag.search.return_value = [
        {"ref": "Isa.40.31", "text": "They shall mount up with wings as eagles"}
    ]

    result = await squad.morning_verse()

    assert "Isa.40.31" in result
    assert "eagles" in result


@pytest.mark.asyncio()
async def test_morning_verse_returns_empty_when_no_results() -> None:
    squad = SpiritualSquad.__new__(SpiritualSquad)
    squad._rag = MagicMock()
    squad._rag.search.return_value = []

    result = await squad.morning_verse()

    assert result == ""
