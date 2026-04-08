"""Tests for mcp_server.server tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── ask_expert tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_routes_to_correct_expert() -> None:
    from mcp_server.server import ask_expert

    mock_result = MagicMock()
    mock_result.content = "Here are the latest AI papers."

    mock_orch = MagicMock()
    mock_orch.running = True
    mock_orch.dispatch = AsyncMock(return_value=mock_result)

    with (
        patch("mcp_server.server._get_orchestrator", return_value=mock_orch),
        patch("router.classifier.IntentRouter") as mock_router_cls,
    ):
        from core.types import Intent, RoutingDecision

        mock_router_cls.return_value.aroute = AsyncMock(
            return_value=(Intent.INTELLIGENCE, 0.95, RoutingDecision.CLOUD)
        )
        result = await ask_expert("intelligence", "What are the latest LLM papers?")

    assert "AI papers" in result
    mock_orch.dispatch.assert_called_once()


@pytest.mark.asyncio()
async def test_rejects_invalid_expert() -> None:
    from mcp_server.server import ask_expert

    result = await ask_expert("nonexistent", "query")
    assert "Unknown expert" in result
    assert "nonexistent" in result


# ── get_memory tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_get_memory_returns_string() -> None:
    from mcp_server.server import get_memory

    with patch("memory.episodic.EpisodicMemory") as mock_cls:
        mock_cls.return_value.get_all.return_value = [
            {"text": "User prefers concise answers"},
            {"text": "User is an ML engineer"},
        ]
        result = await get_memory("ML")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio()
async def test_get_memory_empty() -> None:
    from mcp_server.server import get_memory

    with patch("memory.episodic.EpisodicMemory") as mock_cls:
        mock_cls.return_value.get_all.return_value = []
        result = await get_memory("anything")

    assert "No memory entries" in result


# ── get_skills tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_get_skills_covers_all_intents() -> None:
    from core.types import Intent
    from mcp_server.server import get_skills

    with patch("memory.skill_library.SkillLibrary") as mock_cls:
        mock_cls.return_value.get_top_skills.return_value = ["pattern_a", "pattern_b"]
        for intent in Intent:
            result = await get_skills(intent.value)
            assert isinstance(result, str)
            assert len(result) > 0


# ── get_stats tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_get_stats_returns_summary() -> None:
    from mcp_server.server import get_stats

    with patch("observability.traces.TraceStore") as mock_cls:
        mock_cls.return_value.get_daily_stats.return_value = {
            "total_requests": 42,
            "cache_hits": 10,
            "errors": 0,
            "avg_latency_ms": 350.0,
            "total_tokens_in": 5000,
            "total_tokens_out": 2000,
            "total_cost_usd": 0.12,
            "models_used": "claude-sonnet-4-6",
        }
        result = await get_stats()

    assert "42" in result
    assert "$0.1200" in result
