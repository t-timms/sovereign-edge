"""
Career squad — LangGraph subgraph.

Pipeline:
    START → job_searcher → strategist → END

job_searcher  — Jina web search for live job listings / market data
strategist    — LLM synthesizes search results into career advice / brief
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

_FORMAT_RULES = """\
OUTPUT FORMAT — MANDATORY:
Your responses are delivered via Telegram, which only renders a limited Markdown
subset. You must follow these rules exactly or the output will be unreadable.

ALLOWED:
  *bold* using single asterisks — e.g. *Company Name*
  _italic_ using underscores
  [link text](https://url) — inline links only, never paste bare URLs

FORBIDDEN — these render as literal characters in Telegram:
  **double asterisks** — never use this for bold
  ## headers — never use hash headers
  --- dividers — never use horizontal rules

Structure: one blank line between each job listing.
Keep entries concise: company, title, location, salary, link.\
"""

MORNING_PROMPT = """\
Based on the live job market data above, give a crisp morning career briefing (<= 150 words):
1. One high-value action to take today (specific company to reach out to,
   specific JD to apply for, etc.)
2. One ML/AI market insight from the search results.
Keep it motivating and concrete.\
"""


def build_system_prompt() -> str:
    from core.config import get_settings

    s = get_settings()
    location = s.career_target_location
    roles = s.career_target_roles
    diff_section = (
        f"\nEmphasize the user's differentiators: {s.career_differentiators}."
        if s.career_differentiators
        else ""
    )
    return (
        f"{_FORMAT_RULES}\n\n---\n\n"
        f"You are the Career Intelligence of Sovereign Edge — a world-class career strategist\n"
        f"specializing in {roles} roles in the {location} area.\n\n"
        f"You have access to live search results. When job listings are provided, extract and\n"
        f"present: company name, role title, location, salary range (if shown), and a direct\n"
        f"application link.{diff_section}\n\n"
        f"Be direct, specific, and actionable. When no search results are available, draw on\n"
        f"deep knowledge of the {location} ML/AI job market."
    )


def build_search_queries() -> list[str]:
    from core.config import get_settings

    s = get_settings()
    location = s.career_target_location
    roles = s.career_target_roles.replace(", ", " OR ").replace(",", " OR ")
    return [
        f"{roles} {location} hiring site:linkedin.com OR site:indeed.com",
        f"machine learning engineer jobs {location} remote apply now",
    ]


# ── State ─────────────────────────────────────────────────────────────────────

class CareerState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────
    query: str
    routing: str
    history: list[dict[str, str]]
    is_morning_brief: bool
    # ── Intermediate ──────────────────────────────────────────────────────
    search_results: str
    # ── Outputs ───────────────────────────────────────────────────────────
    response: str
    model_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def _job_searcher(state: CareerState) -> dict[str, Any]:
    """Search for live job listings and market data via Jina (cloud-only)."""
    from core.types import RoutingDecision

    if state["routing"] != RoutingDecision.CLOUD:
        return {"search_results": ""}
    try:
        from search.jina import search as jina_search

        if state["is_morning_brief"]:
            query = build_search_queries()[0]
        else:
            from core.config import get_settings
            location = get_settings().career_target_location
            query = f"{state['query']} ML Engineer AI job {location}"

        results = await jina_search(query, max_results=5)
        logger.info("career_job_searcher chars=%d", len(results))
        return {"search_results": results}
    except Exception:
        logger.warning("career_job_search_failed", exc_info=True)
        return {"search_results": ""}


async def _strategist(state: CareerState) -> dict[str, Any]:
    """Synthesize search results into career advice or a morning brief via LLM."""
    from llm.gateway import get_gateway

    gateway = get_gateway()
    system_prompt = build_system_prompt()

    if state["is_morning_brief"]:
        user_content = (
            f"Live DFW job market results:\n{state['search_results']}\n\n---\n{MORNING_PROMPT}"
            if state["search_results"]
            else MORNING_PROMPT
        )
        max_tokens = 250
    else:
        user_input = f"<user_request>\n{state['query']}\n</user_request>"
        user_content = (
            f"Live search results:\n{state['search_results']}\n\n---\n{user_input}"
            if state["search_results"]
            else user_input
        )
        max_tokens = 1500

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        *state["history"],
        {"role": "user", "content": user_content},
    ]

    result = await gateway.complete(
        messages=messages,
        max_tokens=max_tokens,
        routing=state["routing"],
        squad="career",
    )

    return {
        "response": result["content"],
        "model_used": result.get("model", ""),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "cost_usd": result.get("cost_usd", 0.0),
    }


# ── Graph construction ────────────────────────────────────────────────────────

def _build() -> Any:
    builder: StateGraph = StateGraph(CareerState)

    builder.add_node("job_searcher", _job_searcher)
    builder.add_node("strategist", _strategist)

    builder.add_edge(START, "job_searcher")
    builder.add_edge("job_searcher", "strategist")
    builder.add_edge("strategist", END)

    return builder.compile(name="career_squad")


career_subgraph = _build() if _LANGGRAPH_AVAILABLE else None
