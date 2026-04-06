"""
Sovereign Edge MCP server — 4 tools for Claude Desktop / Claude Code.

Tools
-----
ask_expert    — Dispatch a query to a specific Sovereign Edge expert
get_memory    — Search recent episodic memory entries
get_skills    — Top skill patterns for an intent class
get_stats     — Today's usage and cost stats

Transport: stdio (Claude Desktop) and SSE (daemon mode).

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "sovereign-edge": {
          "command": "python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/home/omnipotence/sovereign-edge"
        }
      }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from core.types import ExpertName, TaskPriority, TaskRequest
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("sovereign-edge")

# ── Lazy orchestrator ──────────────────────────────────────────────────────────

_orchestrator: Any = None
_VALID_EXPERTS = {e.value for e in ExpertName}


def _get_orchestrator() -> object:
    global _orchestrator
    if _orchestrator is None:
        from career.expert import CareerExpert
        from creative.expert import CreativeExpert
        from goals.expert import GoalExpert
        from intelligence.expert import IntelligenceExpert
        from orchestrator.main import Orchestrator
        from spiritual.expert import SpiritualExpert

        orch = Orchestrator()
        for expert in (
            SpiritualExpert(),
            CareerExpert(),
            IntelligenceExpert(),
            CreativeExpert(),
            GoalExpert(),
        ):
            orch.register(expert)
        _orchestrator = orch
    return _orchestrator


# ── Tools ──────────────────────────────────────────────────────────────────────


@mcp.tool()
async def ask_expert(expert: str, query: str) -> str:
    """Dispatch a query to a Sovereign Edge expert.

    Parameters
    ----------
    expert:
        One of: spiritual, career, intelligence, creative, goals
    query:
        The question or task to send to the expert.
    """
    if expert not in _VALID_EXPERTS:
        valid = ", ".join(sorted(_VALID_EXPERTS))
        return f"Unknown expert '{expert}'. Valid options: {valid}"

    try:
        from router.classifier import IntentRouter

        orch = _get_orchestrator()
        router = IntentRouter()
        intent, _confidence, routing = await router.aroute(query)

        request = TaskRequest(
            content=query,
            intent=intent,
            priority=TaskPriority.HIGH,
            routing=routing,
            context={"chat_id": "mcp"},
        )

        if not orch.running:
            await orch.start()

        result = await orch.dispatch(request)
        return result.content
    except Exception:
        logger.error("mcp_ask_expert_failed expert=%s", expert, exc_info=True)
        return f"Error dispatching to {expert}. Check logs for details."


@mcp.tool()
async def get_memory(query: str) -> str:
    """Search recent episodic memory entries.

    Parameters
    ----------
    query:
        Search term to filter memory entries (case-insensitive substring match).
    """
    try:
        from memory.episodic import EpisodicMemory

        entries = EpisodicMemory().get_all()
        if not entries:
            return "No episodic memory entries found."

        q = query.lower()
        matching = [e for e in entries if q in str(e).lower()] if q else entries
        if not matching:
            return f"No memory entries matching '{query}'."

        lines = [f"- {e}" for e in matching[:20]]
        return "\n".join(lines)
    except Exception:
        logger.error("mcp_get_memory_failed", exc_info=True)
        return "Memory unavailable."


@mcp.tool()
async def get_skills(intent: str) -> str:
    """Top skill patterns for an intent class.

    Parameters
    ----------
    intent:
        One of: SPIRITUAL, CAREER, INTELLIGENCE, CREATIVE, GOALS, GENERAL
    """
    try:
        from memory.skill_library import SkillLibrary

        lib = SkillLibrary()
        skills = lib.get_top_skills(intent.upper())
        if not skills:
            return f"No skills recorded yet for intent '{intent}'."
        return "\n".join(f"- {s}" for s in skills)
    except Exception:
        logger.error("mcp_get_skills_failed intent=%s", intent, exc_info=True)
        return "Skills unavailable."


@mcp.tool()
async def get_stats() -> str:
    """Today's usage and cost stats from the Sovereign Edge orchestrator."""
    try:
        from observability.traces import TraceStore

        stats = TraceStore().get_daily_stats()
        if not stats:
            return "No stats recorded today yet."

        return (
            f"Requests: {stats.get('total_requests', 0)}\n"
            f"Cache hits: {stats.get('cache_hits', 0)}\n"
            f"Errors: {stats.get('errors', 0)}\n"
            f"Avg latency: {stats.get('avg_latency_ms', 0.0):.0f}ms\n"
            f"Tokens in: {stats.get('total_tokens_in', 0):,}\n"
            f"Tokens out: {stats.get('total_tokens_out', 0):,}\n"
            f"Total cost: ${stats.get('total_cost_usd', 0.0):.4f}\n"
            f"Models used: {stats.get('models_used') or 'none'}"
        )
    except Exception:
        logger.error("mcp_get_stats_failed", exc_info=True)
        return "Stats unavailable."


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    # stdio transport for Claude Desktop; use mcp.run(transport="sse") for daemon
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
