"""
Career expert — LangGraph subgraph.

Pipeline:
    START → job_searcher → strategist → END

job_searcher  — Jina web search for live job listings / market data
strategist    — LLM synthesizes search results into career advice / brief
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

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

# ── Structured output models ───────────────────────────────────────────────────

_DFW_CITIES: frozenset[str] = frozenset({
    "dallas", "fort worth", "plano", "irving", "frisco", "allen", "mckinney",
    "richardson", "arlington", "southlake", "addison", "carrollton", "garland",
    "lewisville", "denton", "grand prairie", "mesquite", "rowlett", "mansfield",
    "dfw", "dallas-fort worth", "dallas fort worth",
})


class JobListing(BaseModel):
    """A single verified DFW job listing."""

    company: str = Field(description="Company name")
    title: str = Field(description="Exact job title")
    city: str = Field(description="City where the job is located")
    state: str = Field(default="TX", description="US state abbreviation")
    work_mode: str = Field(
        default="onsite",
        description="One of: onsite, hybrid, remote",
    )
    salary_range: str = Field(default="", description="Salary range if listed, else empty string")
    apply_url: str = Field(description="Direct application URL")

    @field_validator("city", mode="before")
    @classmethod
    def normalize_city(cls, v: str) -> str:
        return v.strip().title()

    @property
    def is_dfw_eligible(self) -> bool:
        """True if the job is in a DFW city or explicitly remote/hybrid."""
        city_lower = self.city.lower()
        return (
            any(dfw_city in city_lower for dfw_city in _DFW_CITIES)
            or self.work_mode in ("remote", "hybrid")
        )


class JobListingResponse(BaseModel):
    """Structured response from the career strategist."""

    jobs: list[JobListing] = Field(description="Job listings found, DFW-eligible only")
    action_today: str = Field(description="One concrete action to take today (1 sentence)")


def format_job_listings(response: JobListingResponse) -> str:
    """Render a JobListingResponse as Telegram-safe numbered markdown."""
    eligible = [j for j in response.jobs if j.is_dfw_eligible]
    if not eligible:
        action = f"\n\n*Action today:* {response.action_today}" if response.action_today else ""
        return f"No DFW-eligible listings found in today's search.{action}"

    lines: list[str] = []
    for i, job in enumerate(eligible, 1):
        salary = f" | {job.salary_range}" if job.salary_range else ""
        mode = f" ({job.work_mode.title()})" if job.work_mode != "onsite" else ""
        lines.append(f"{i}. *{job.company}* — {job.title}")
        lines.append(f"   _{job.city}, {job.state}{mode}_{salary}")
        lines.append(f"   [Apply →]({job.apply_url})")
        lines.append("")

    if response.action_today:
        lines.append(f"*Action today:* {response.action_today}")

    return "\n".join(lines).strip()


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

JOB LIST FORMAT — each entry must follow this exact template:
  1. *Company Name* — Role Title
     _City, TX_ | Salary: $XXXk–$XXXk (if listed)
     [Apply →](https://direct-link)

  2. *Company Name* — Role Title
     ...

One blank line between entries. Never use bullet points for job listings.
Skip any job not located in the configured target area (remote-eligible is OK).\
"""

MORNING_PROMPT = """\
Based on the live job market data above, list the top DFW jobs found today using the
job list format above. Then add one sentence: a high-value action to take today.
Max 5 job listings. Only include jobs in the configured location or remote-eligible.\
"""


def build_system_prompt() -> str:
    from core.config import get_settings

    s = get_settings()
    location = s.career_target_location
    cities = s.career_target_cities
    roles = s.career_target_roles
    diff_section = (
        f"\nEmphasize the user's differentiators: {s.career_differentiators}."
        if s.career_differentiators
        else ""
    )
    return (
        f"{_FORMAT_RULES}\n\n---\n\n"
        f"You are the Career Intelligence of Sovereign Edge — a world-class career strategist\n"
        f"specializing in {roles} roles.\n\n"
        f"TARGET AREA: {location}. Only include jobs located in one of these cities: {cities},\n"
        f"OR jobs explicitly marked remote/hybrid that are open to {location} candidates.\n"
        f"HARD RULE: Do not list jobs in other cities (NYC, SF, Seattle, etc.) unless remote.\n\n"
        f"You have access to live search results. For each job listing extract and present:\n"
        f"company name, role title, city/state, salary (if shown), direct application link.{diff_section}\n\n"
        f"When no search results are available, share 2-3 known {location} ML/AI employers\n"
        f"actively hiring and the best way to apply."
    )


def build_search_queries() -> list[str]:
    from core.config import get_settings

    s = get_settings()
    location = s.career_target_location
    roles = s.career_target_roles.replace(", ", " OR ").replace(",", " OR ")
    # Use specific DFW city names for tighter location filtering
    return [
        f'("{roles}") ("{location}" OR "Dallas" OR "Plano" OR "Irving" OR "Frisco") site:linkedin.com OR site:indeed.com',
        f'machine learning engineer AI engineer "{location}" OR "Dallas TX" OR "Plano TX" jobs -"New York" -"San Francisco" -"Seattle"',
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
            s = get_settings()
            location = s.career_target_location
            # Tighten location filter: exclude known non-DFW metros
            query = (
                f'{state["query"]} ML Engineer AI job "{location}" OR "Dallas" OR "Plano" OR "Irving" '
                f'-"New York" -"San Francisco" -"Chicago" -"Seattle" -"Austin"'
            )

        results = await jina_search(query, max_results=5)
        logger.info("career_job_searcher chars=%d", len(results))
        return {"search_results": results}
    except Exception:
        logger.warning("career_job_search_failed", exc_info=True)
        return {"search_results": ""}


async def _strategist(state: CareerState) -> dict[str, Any]:
    """Synthesize search results into career advice or a morning brief via LLM.

    Attempts structured output (JobListingResponse) first for guaranteed format.
    Falls back to unstructured complete() if structured fails.
    """
    from llm.gateway import get_gateway

    gateway = get_gateway()
    system_prompt = build_system_prompt()

    if state["is_morning_brief"]:
        user_content = (
            f"Live DFW job market results:\n{state['search_results']}\n\n---\n{MORNING_PROMPT}"
            if state["search_results"]
            else MORNING_PROMPT
        )
        max_tokens = 800
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

    # ── Structured path (instructor + Pydantic) ────────────────────────────
    structured = await gateway.complete_structured(
        messages=messages,
        response_model=JobListingResponse,
        routing=state["routing"],
        max_tokens=max_tokens,
        temperature=0.2,
        expert="career",
    )
    if structured is not None:
        logger.info(
            "career_strategist_structured jobs=%d",
            len([j for j in structured.jobs if j.is_dfw_eligible]),
        )
        return {
            "response": format_job_listings(structured),
            "model_used": "structured",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }

    # ── Unstructured fallback ──────────────────────────────────────────────
    logger.warning("career_strategist_structured_failed — falling back to unstructured")
    result = await gateway.complete(
        messages=messages,
        max_tokens=max_tokens,
        routing=state["routing"],
        expert="career",
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

    return builder.compile(name="career_expert")


career_subgraph = _build() if _LANGGRAPH_AVAILABLE else None
