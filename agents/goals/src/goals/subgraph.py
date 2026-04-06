"""
Goals expert — LangGraph subgraph.

Pipeline:
    START → goal_router → [goal_writer | goal_reader] → llm_formatter → END

goal_router   — parses action from user query (add/update/list/complete)
goal_writer   — mutates GoalStore (add/update/complete)
goal_reader   — reads GoalStore (list)
llm_formatter — generates natural language response
"""

from __future__ import annotations

import logging
import re
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

# ── State ─────────────────────────────────────────────────────────────────────


class GoalState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────
    query: str
    routing: str
    # ── Intermediate ──────────────────────────────────────────────────────
    action: str  # "add" | "update" | "list" | "complete" | "unknown"
    goal_id: int | None
    title: str
    description: str
    target_date: str | None
    progress: int
    store_result: str  # human-readable result from GoalStore operation
    # ── Outputs ───────────────────────────────────────────────────────────
    response: str
    model_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ── Nodes ─────────────────────────────────────────────────────────────────────

_ADD_RE = re.compile(
    r"\b(?:add|create|new|track)\b.*?goal[:\s]+(.+?)(?:\s+by\s+(\S+))?$",
    re.IGNORECASE | re.DOTALL,
)
_UPDATE_RE = re.compile(
    r"\b(?:update|set|mark)\b.*?(?:goal\s*)?#?(\d+).*?(\d{1,3})\s*%",
    re.IGNORECASE,
)
_COMPLETE_RE = re.compile(
    r"\b(?:complete|finish|done|close)\b.*?(?:goal\s*)?#?(\d+)",
    re.IGNORECASE,
)
_LIST_RE = re.compile(r"\b(?:list|show|what|my|all)\b.*?goal", re.IGNORECASE)


def _goal_router(state: GoalState) -> dict[str, Any]:
    """Parse the user query to determine the goal operation."""
    q = state["query"]

    m = _ADD_RE.search(q)
    if m:
        return {
            "action": "add",
            "title": m.group(1).strip().rstrip("."),
            "target_date": m.group(2) if m.lastindex and m.lastindex >= 2 else None,
        }

    m = _UPDATE_RE.search(q)
    if m:
        return {"action": "update", "goal_id": int(m.group(1)), "progress": int(m.group(2))}

    m = _COMPLETE_RE.search(q)
    if m:
        return {"action": "complete", "goal_id": int(m.group(1))}

    if _LIST_RE.search(q):
        return {"action": "list"}

    return {"action": "unknown"}


def _goal_writer(state: GoalState) -> dict[str, Any]:
    """Mutate GoalStore based on parsed action."""
    from goals.store import GoalStore

    store = GoalStore()
    action = state["action"]

    if action == "add":
        goal_id = store.add_goal(
            title=state["title"],
            description=state["description"],
            target_date=state["target_date"],
        )
        return {"store_result": f"Goal #{goal_id} added: {state['title']}"}

    if action == "update":
        gid = state["goal_id"]
        if gid is None:
            return {"store_result": "Error: no goal ID specified"}
        store.update_progress(gid, state["progress"])
        return {"store_result": f"Goal #{gid} updated to {state['progress']}%"}

    if action == "complete":
        gid = state["goal_id"]
        if gid is None:
            return {"store_result": "Error: no goal ID specified"}
        store.mark_complete(gid)
        return {"store_result": f"Goal #{gid} marked complete"}

    return {"store_result": ""}


def _goal_reader(state: GoalState) -> dict[str, Any]:
    """Read active goals from GoalStore."""
    from goals.store import GoalStore

    store = GoalStore()
    goals = store.list_goals(status="active")
    if not goals:
        return {"store_result": "No active goals."}

    lines = []
    for g in goals:
        due = f" (due {g.target_date})" if g.target_date else ""
        lines.append(f"#{g.id}: {g.title} — {g.progress_pct}%{due}")
    return {"store_result": "\n".join(lines)}


async def _llm_formatter(state: GoalState) -> dict[str, Any]:
    """Generate a natural language response from the store result."""
    from llm.gateway import get_gateway

    gateway = get_gateway()
    store_result = state["store_result"]
    query = state["query"]

    if not store_result or state["action"] == "unknown":
        # No LLM call needed for unknown — just explain usage
        usage = (
            "I can help you manage your goals. Try:\n"
            "• *add goal: [title] by [date]* — track a new goal\n"
            "• *update goal #1 to 50%* — update progress\n"
            "• *complete goal #2* — mark done\n"
            "• *list goals* — see all active goals"
        )
        return {
            "response": usage,
            "model_used": "none",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Goals Intelligence of Sovereign Edge — a focused personal coach.\n"
                "Respond to goal management actions concisely and motivationally.\n"
                "Use Telegram-safe Markdown: *bold*, _italic_, [link](url). No ## headers.\n"
                "Keep responses under 150 words."
            ),
        },
        {
            "role": "user",
            "content": f"User request: {query}\n\nResult: {store_result}",
        },
    ]

    result = await gateway.complete(
        messages=messages,
        max_tokens=200,
        routing=state["routing"],
        expert="goals",
    )
    return {
        "response": result["content"],
        "model_used": result.get("model", ""),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "cost_usd": result.get("cost_usd", 0.0),
    }


def _route_action(state: GoalState) -> str:
    """Conditional edge: write vs read vs format."""
    action = state["action"]
    if action in ("add", "update", "complete"):
        return "write"
    if action == "list":
        return "read"
    return "format"  # unknown → skip to formatter


# ── Graph construction ────────────────────────────────────────────────────────


def _build() -> Any:  # noqa: ANN401
    builder: StateGraph = StateGraph(GoalState)

    builder.add_node("goal_router", _goal_router)
    builder.add_node("goal_writer", _goal_writer)
    builder.add_node("goal_reader", _goal_reader)
    builder.add_node("llm_formatter", _llm_formatter)

    builder.add_edge(START, "goal_router")
    builder.add_conditional_edges(
        "goal_router",
        _route_action,
        {"write": "goal_writer", "read": "goal_reader", "format": "llm_formatter"},
    )
    builder.add_edge("goal_writer", "llm_formatter")
    builder.add_edge("goal_reader", "llm_formatter")
    builder.add_edge("llm_formatter", END)

    return builder.compile(name="goals_expert")


goals_subgraph = _build() if _LANGGRAPH_AVAILABLE else None
