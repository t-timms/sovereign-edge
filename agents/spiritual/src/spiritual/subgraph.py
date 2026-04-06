"""
Spiritual expert — LangGraph subgraph.

Pipeline:
    START → scripture_fetcher → theologian → END

scripture_fetcher  — Bible API lookup (free, no auth)
theologian         — LLM generates devotional / scripture reflection / prayer
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
  *bold* using single asterisks — e.g. *Reflection:*
  _italic_ using underscores — use for all scripture quotations
  [link text](https://url) — inline links only, never paste bare URLs

FORBIDDEN — these render as literal characters in Telegram:
  **double asterisks** — never use this for bold
  ## headers — never use hash headers
  --- dividers — never use horizontal rules

Structure: separate the verse, reflection, and prayer with one blank line each.
Use *bold labels* like *Reflection:* and *Prayer:* as section markers.
Keep paragraphs to 2-3 sentences. Warm and readable, never dense.

---

You are the Spiritual Intelligence of Sovereign Edge — a contemplative guide
rooted in Christian faith.

You have access to live Bible verse lookups. When scripture is provided in the
context, quote it exactly as retrieved and cite book, chapter, and verse. Help
with scripture study, prayer composition, theological questions, and daily
devotionals. Respond with depth, warmth, and scriptural grounding.

Format scripture quotes in italics with full citation
(e.g., _"For God so loved the world..."_ — John 3:16 KJV).\
"""

DEVOTIONAL_PROMPT = """\
Using the scripture verse above as your anchor, write a brief morning devotional:
1. Quote the verse exactly.
2. 2-3 sentences of reflection connecting it to daily life.
3. A one-sentence prayer.
Keep it under 120 words. Warm and personal in tone.\
"""


# ── State ─────────────────────────────────────────────────────────────────────


class SpiritualState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────
    query: str
    routing: str
    history: list[dict[str, str]]
    is_morning_brief: bool
    # ── Intermediate ──────────────────────────────────────────────────────
    scripture: str  # formatted verse text from Bible API
    # ── Outputs ───────────────────────────────────────────────────────────
    response: str
    model_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ── Nodes ─────────────────────────────────────────────────────────────────────


async def _scripture_fetcher(state: SpiritualState) -> dict[str, Any]:
    """Fetch a Bible verse — specific reference if found in query, else random."""
    from core.types import RoutingDecision

    if state["routing"] != RoutingDecision.CLOUD:
        return {"scripture": ""}
    try:
        from search.bible import extract_reference, format_verse, lookup, random_verse

        if state["is_morning_brief"]:
            verse = await random_verse()
        else:
            ref = extract_reference(state["query"])
            verse = await lookup(ref) if ref else await random_verse()

        formatted = format_verse(verse)
        logger.info("spiritual_scripture_fetched ref=%s", formatted[:60] if formatted else "none")
        return {"scripture": formatted or ""}
    except Exception:
        logger.warning("spiritual_scripture_fetch_failed", exc_info=True)
        return {"scripture": ""}


async def _theologian(state: SpiritualState) -> dict[str, Any]:
    """Generate a devotional or scripture response via LLM."""
    from llm.gateway import get_gateway

    gateway = get_gateway()

    scripture_context = f"Scripture:\n{state['scripture']}" if state["scripture"] else ""

    if state["is_morning_brief"]:
        user_content = (
            f"{scripture_context}\n\n---\n{DEVOTIONAL_PROMPT}"
            if scripture_context
            else (
                "Generate a brief morning devotional with a scripture verse, "
                "2-3 sentences of reflection, and a one-sentence prayer. Under 120 words."
            )
        )
        max_tokens = 300
    else:
        user_input = f"<user_request>\n{state['query']}\n</user_request>"
        user_content = (
            f"{scripture_context}\n\n---\n{user_input}" if scripture_context else user_input
        )
        max_tokens = 1024

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state["history"],
        {"role": "user", "content": user_content},
    ]

    result = await gateway.complete(
        messages=messages,
        max_tokens=max_tokens,
        routing=state["routing"],
        expert="spiritual",
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
    builder: StateGraph = StateGraph(SpiritualState)

    builder.add_node("scripture_fetcher", _scripture_fetcher)
    builder.add_node("theologian", _theologian)

    builder.add_edge(START, "scripture_fetcher")
    builder.add_edge("scripture_fetcher", "theologian")
    builder.add_edge("theologian", END)

    return builder.compile(name="spiritual_expert")


spiritual_subgraph = _build() if _LANGGRAPH_AVAILABLE else None
