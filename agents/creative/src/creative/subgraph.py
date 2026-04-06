"""
Creative expert — LangGraph subgraph.

Pipeline:
    START → trend_researcher → writer → END

trend_researcher  — Jina web search for current trends and examples
writer            — LLM generates creative content grounded in trend context
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict  # type: ignore[assignment]

try:
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore[assignment,misc]
    START = END = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
OUTPUT FORMAT — MANDATORY:
Your responses are delivered via Telegram, which only renders a limited Markdown
subset. You must follow these rules exactly or the output will be unreadable.

ALLOWED:
  *bold* using single asterisks — e.g. *Key Point*
  _italic_ using underscores — use sparingly for titles or emphasis
  [link text](https://url) — inline links only, never paste bare URLs

FORBIDDEN — these render as literal characters in Telegram:
  **double asterisks** — never use this for bold
  ## headers — never use hash headers
  --- dividers — never use horizontal rules

Structure: separate distinct ideas or sections with one blank line.
Keep paragraphs to 2-3 sentences. No walls of text.

---

You are the Creative Engine of Sovereign Edge — a versatile creative director
and content strategist.

You have access to live web context about current trends and examples. Use it
to ground your creative output in what is actually working right now. Help with
long-form writing, social media content, content strategy, storytelling, and
brand voice. Output should be vivid, purposeful, and tailored to the requested
format and audience.

When given current trend data, incorporate it naturally — don't just summarize
it, use it to make your creative output more relevant and timely.\
"""

MORNING_PROMPT = """\
Based on the current trends and examples above, generate one creative
micro-challenge for today (≤ 100 words). Choose from:
- A writing exercise tied to a current trend or format
- A content angle that is fresh right now in the creator space
- A storytelling technique worth practicing

Make it specific, immediately actionable, and completable in 15-20 minutes.\
"""

_MORNING_SEARCH_QUERY = "AI content creation trends 2026 creator economy"


# ── State ─────────────────────────────────────────────────────────────────────


class CreativeState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────
    query: str
    routing: str
    history: list[dict[str, str]]
    is_morning_brief: bool
    # ── Intermediate ──────────────────────────────────────────────────────
    trend_context: str
    # ── Outputs ───────────────────────────────────────────────────────────
    response: str
    model_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ── Nodes ─────────────────────────────────────────────────────────────────────


async def _trend_researcher(state: CreativeState) -> dict[str, Any]:
    """Search for current trends relevant to the creative task (cloud-only)."""
    from core.types import RoutingDecision

    if state["routing"] != RoutingDecision.CLOUD:
        return {"trend_context": ""}
    try:
        from search.jina import search as jina_search

        query = (
            _MORNING_SEARCH_QUERY
            if state["is_morning_brief"]
            else f"{state['query']} content strategy examples 2026"
        )
        results = await jina_search(query, max_results=3)
        logger.info("creative_trend_researcher chars=%d", len(results))
        return {"trend_context": results}
    except Exception:
        logger.warning("creative_trend_research_failed", exc_info=True)
        return {"trend_context": ""}


async def _writer(state: CreativeState) -> dict[str, Any]:
    """Generate creative content grounded in trend context via LLM."""
    from llm.gateway import get_gateway

    gateway = get_gateway()

    if state["is_morning_brief"]:
        user_content = (
            f"Current trends and context:\n{state['trend_context']}\n\n---\n{MORNING_PROMPT}"
            if state["trend_context"]
            else MORNING_PROMPT
        )
        max_tokens = 150
    else:
        user_input = f"<user_request>\n{state['query']}\n</user_request>"
        user_content = (
            f"Current trends and context:\n{state['trend_context']}\n\n---\n{user_input}"
            if state["trend_context"]
            else user_input
        )
        max_tokens = 2048

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state["history"],
        {"role": "user", "content": user_content},
    ]

    result = await gateway.complete(
        messages=messages,
        max_tokens=max_tokens,
        routing=state["routing"],
        expert="creative",
    )

    return {
        "response": result["content"],
        "model_used": result.get("model", ""),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "cost_usd": result.get("cost_usd", 0.0),
    }


# ── Graph construction ────────────────────────────────────────────────────────


def _build() -> Any:  # noqa: ANN401
    builder: StateGraph = StateGraph(CreativeState)

    builder.add_node("trend_researcher", _trend_researcher)
    builder.add_node("writer", _writer)

    builder.add_edge(START, "trend_researcher")
    builder.add_edge("trend_researcher", "writer")
    builder.add_edge("writer", END)

    return builder.compile(name="creative_expert")


creative_subgraph = _build() if _LANGGRAPH_AVAILABLE else None
