"""
Career expert — LangGraph subgraph.

Pipeline:
    START → job_searcher → strategist → END

job_searcher  — Multi-source job fetching (The Muse + Remotive + Adzuna + Jina),
                SQLite deduplication (7-day window), resume skill extraction.
strategist    — LLM synthesizes new listings + resume profile into career advice/brief.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any

import httpx
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

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_SEARCH_CHARS = 12_000  # merged cap for all sources combined

# ── Structured output models ───────────────────────────────────────────────────

_DFW_CITIES: frozenset[str] = frozenset(
    {
        "dallas",
        "fort worth",
        "plano",
        "irving",
        "frisco",
        "allen",
        "mckinney",
        "richardson",
        "arlington",
        "southlake",
        "addison",
        "carrollton",
        "garland",
        "lewisville",
        "denton",
        "grand prairie",
        "mesquite",
        "rowlett",
        "mansfield",
        "dfw",
        "dallas-fort worth",
        "dallas fort worth",
    }
)


class JobListing(BaseModel):
    """A single verified DFW job listing with A-F fit score."""

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
    score: float = Field(
        default=0.0,
        description=(
            "Fit score 0.0-5.0. Weighted across: "
            "A) Role archetype match (ML/AI/LLM/Agentic/LLMOps) "
            "B) Skill alignment with candidate profile "
            "C) Seniority level fit (mid/senior) "
            "D) Location/remote eligibility. "
            "5.0 = perfect match, 4.0+ = strong, 3.0-4.0 = reasonable, <3.0 = weak."
        ),
    )
    score_rationale: str = Field(
        default="",
        description="One sentence explaining the score — what drives it up or down.",
    )

    @field_validator("city", mode="before")
    @classmethod
    def normalize_city(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(5.0, float(v)))

    @property
    def is_dfw_eligible(self) -> bool:
        """True if the job is in a DFW city or explicitly remote/hybrid."""
        city_lower = self.city.lower()
        return any(dfw_city in city_lower for dfw_city in _DFW_CITIES) or self.work_mode in (
            "remote",
            "hybrid",
        )

    @property
    def score_stars(self) -> str:
        """Star rating string for Telegram display."""
        filled = round(self.score)
        return "★" * filled + "☆" * (5 - filled)


class JobListingResponse(BaseModel):
    """Structured response from the career strategist."""

    jobs: list[JobListing] = Field(description="Job listings found, DFW-eligible only")
    action_today: str = Field(description="One concrete action to take today (1 sentence)")
    skill_gap: str = Field(
        default="",
        description="One skill gap observed between job requirements and candidate profile "
        "(1 sentence). Empty if no gap is apparent.",
    )


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
        score_str = f" {job.score_stars} {job.score:.1f}/5" if job.score > 0 else ""
        lines.append(f"{i}.{score_str} *{job.company}* — {job.title}")
        lines.append(f"   _{job.city}, {job.state}{mode}_{salary}")
        if job.score_rationale:
            lines.append(f"   _{job.score_rationale}_")
        lines.append(f"   [Apply →]({job.apply_url})")
        lines.append("")

    if response.action_today:
        lines.append(f"*Action today:* {response.action_today}")
    if response.skill_gap:
        lines.append(f"*Skill gap noted:* {response.skill_gap}")

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
     _City, TX_ | Salary: $XXXk-$XXXk (if listed)
     [Apply →](https://direct-link)

  2. *Company Name* — Role Title
     ...

One blank line between entries. Never use bullet points for job listings.
Skip any job not located in the configured target area (remote-eligible is OK).\
"""

MORNING_PROMPT = """\
Based on the live job market data above, list the top DFW jobs found today using the
job list format above. Then add one sentence: a high-value action to take today.
Max 5 job listings. Only include jobs in the configured location or remote-eligible.
If a candidate skill profile was provided, note any significant skill gap in one sentence.\
"""


def build_system_prompt(resume_context: str = "") -> str:
    """Build the career strategist system prompt.

    Injects resume skill profile when provided so the LLM can match jobs
    to the candidate's actual skills and flag gaps.
    """
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
    resume_section = f"\n\n{resume_context}" if resume_context else ""

    scoring_rules = """\

JOB FIT SCORING — MANDATORY for every listing:
Score each job 0.0-5.0 across four dimensions (weight in parentheses):
  A. Archetype match (30%): Does the role align with ML/AI/LLM engineering, LLMOps,
     or agentic systems? Exact title match = 5, adjacent = 3, stretch = 1.
  B. Skill alignment (30%): How well do the JD's required skills match the candidate
     profile above? Count matched skills as a fraction of required skills.
  C. Level fit (20%): Mid/ML Engineer II/Senior = 5, Junior = 2, Principal/Staff = 3
     (stretch). Roles requiring 5+ years where candidate has 2 = 2.5.
  D. Location (20%): DFW on-site or hybrid = 5, US remote = 4.5, other = 1.
Final score = (A*0.30) + (B*0.30) + (C*0.20) + (D*0.20). Round to 1 decimal.
Also write one sentence in score_rationale explaining the primary driver.
STRONG RECOMMENDATION: Only surface jobs with score ≥ 3.5. Flag 4.5+ as priority.\
"""

    return (
        f"{_FORMAT_RULES}\n\n---\n\n"
        f"You are the Career Intelligence of Sovereign Edge — a world-class career strategist\n"
        f"specializing in {roles} roles.\n\n"
        f"TARGET AREA: {location}. Only include jobs in: {cities},\n"
        f"OR jobs explicitly marked remote/hybrid open to {location} candidates.\n"
        f"HARD RULE: Do not list jobs in other cities (NYC, SF, Seattle, etc.) unless remote.\n\n"
        f"You have access to live search results. For each job listing extract and present:\n"
        f"company name, role title, city/state, salary (if shown), and an application link."
        f"{diff_section}\n\n"
        f"LINK PRIORITY — LinkedIn and Indeed job URLs expire within days. Use this order:\n"
        f"  1. Company careers page (careers.company.com) — most stable\n"
        f"  2. Direct ATS link (greenhouse.io, lever.co, workday.com, icims.com)\n"
        f"  3. LinkedIn or Indeed URL — only if no stable link is available\n\n"
        f"CRITICAL: Only list jobs explicitly present in the LIVE SEARCH RESULTS above. "
        f"Never generate, guess, or fabricate job listings, companies, or URLs. "
        f"If no search results are provided, state that clearly."
        f"{resume_section}"
        f"\n\n{scoring_rules}"
    )


def build_search_queries() -> list[str]:
    """Build Jina search queries — supplementary to structured API sources."""
    from core.config import get_settings

    s = get_settings()
    location = s.career_target_location
    roles = s.career_target_roles
    today = datetime.date.today()
    return [
        f"{roles} jobs hiring Dallas Fort Worth Texas {today.year}",
        f"machine learning AI engineer positions {location} {today.year}",
    ]


# ── State ─────────────────────────────────────────────────────────────────────


class CareerState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────
    query: str
    routing: str
    history: list[dict[str, str]]
    is_morning_brief: bool
    # ── Intermediate ──────────────────────────────────────────────────────
    search_results: str  # Formatted context from all sources (deduped)
    new_job_count: int  # Count of new jobs found (before LLM structuring)
    resume_context: str  # Skill profile extracted from PDF resumes
    # ── Outputs ───────────────────────────────────────────────────────────
    response: str
    model_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ── Nodes ─────────────────────────────────────────────────────────────────────


async def _job_searcher(state: CareerState) -> dict[str, Any]:
    """Multi-source job fetcher with deduplication and resume intelligence.

    Sources (all free):
      - The Muse API — structured job listings, no auth
      - Remotive API — remote ML/AI jobs, no auth
      - Adzuna API   — comprehensive job DB, free 50 req/day (SE_ADZUNA_APP_ID/KEY)
      - Jina search  — supplementary web search

    Deduplication:
      SQLite job store filters jobs seen in the last SE_CAREER_DEDUP_WINDOW_DAYS days.
      Only new jobs are surfaced. Seen jobs are marked immediately.

    Resume intelligence:
      Parses PDFs from SE_CAREER_RESUME_PATH via pypdf — extracts skills for LLM injection.
    """
    from core.config import get_settings
    from core.types import RoutingDecision

    if state["routing"] != RoutingDecision.CLOUD:
        return {"search_results": "", "new_job_count": 0, "resume_context": ""}

    s = get_settings()

    try:
        from search.job_store import JobStore
        from search.jobs import fetch_all_sources, format_raw_listings
        from search.resume_intel import build_resume_profile

        # ── All sources run in parallel ────────────────────────────────────
        if state["is_morning_brief"]:
            jina_queries = build_search_queries()
        else:
            year = datetime.date.today().year
            jina_queries = [
                f"ML Engineer AI jobs {state['query']} Dallas Fort Worth Texas {year}",
                f"machine learning engineer jobs hiring {s.career_target_location} {year}",
            ]

        from search.jina import search as jina_search

        raw_listings_coro = fetch_all_sources(
            adzuna_app_id=s.adzuna_app_id.get_secret_value(),
            adzuna_app_key=s.adzuna_app_key.get_secret_value(),
        )
        resume_coro = asyncio.to_thread(build_resume_profile, s.career_resume_path)
        jina_coros = [jina_search(q, max_results=5) for q in jina_queries]

        gathered = await asyncio.gather(
            raw_listings_coro,
            resume_coro,
            *jina_coros,
            return_exceptions=True,
        )

        raw_listings = gathered[0] if isinstance(gathered[0], list) else []
        resume_profile = gathered[1] if not isinstance(gathered[1], Exception) else None
        jina_texts = [r for r in gathered[2:] if isinstance(r, str) and r]

        # ── Deduplication (morning brief only) ────────────────────────────
        # Morning briefs dedup to avoid repeating yesterday's jobs.
        # On-demand queries show all available positions — the user is
        # explicitly asking for what's out there.
        db_path = s.career_job_db_path if s.career_job_db_path else (s.ssd_root / "jobs.db")
        store = JobStore(db_path)
        if state["is_morning_brief"]:
            new_listings = store.filter_new(
                raw_listings, dedup_window_days=s.career_dedup_window_days
            )
            store.mark_seen(new_listings)
        else:
            new_listings = raw_listings

        # ── Build combined context for LLM ─────────────────────────────────
        parts: list[str] = []
        if new_listings:
            parts.append(format_raw_listings(new_listings))
        jina_merged = "\n\n".join(jina_texts)
        if jina_merged:
            parts.append(f"Additional web search results:\n{jina_merged[:6_000]}")

        search_results = "\n\n".join(parts)[:_MAX_SEARCH_CHARS]

        resume_context = ""
        if resume_profile is not None and not isinstance(resume_profile, Exception):
            resume_context = resume_profile.to_context_string()

        logger.info(
            "career_job_searcher raw=%d new=%d jina_chars=%d resume_skills=%d",
            len(raw_listings),
            len(new_listings),
            len(jina_merged),
            len(resume_profile.all_skills_flat) if resume_profile else 0,
        )
        return {
            "search_results": search_results,
            "new_job_count": len(new_listings),
            "resume_context": resume_context,
        }

    except Exception:
        logger.warning("career_job_search_failed", exc_info=True)
        return {"search_results": "", "new_job_count": 0, "resume_context": ""}


async def _check_url_live(url: str, client: httpx.AsyncClient) -> bool:
    """HEAD-check a URL; returns True if non-4xx (or on any network error — fail open)."""
    if not url or not url.startswith("http"):
        return False
    try:
        resp = await client.head(url, follow_redirects=True)
        return resp.status_code < 400
    except httpx.RequestError:
        return True  # Fail open: keep listing if we can't reach the URL


async def _validate_listings(listings: list[JobListing]) -> list[JobListing]:
    """Drop listings whose apply_url returns a 4xx HTTP response.

    Uses a single shared AsyncClient for all checks. Fails open (keeps listing)
    on network errors so transient connectivity issues don't blank the response.
    """
    if not listings:
        return listings
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            checks = await asyncio.gather(
                *[_check_url_live(j.apply_url, client) for j in listings],
                return_exceptions=True,
            )
        live = [j for j, ok in zip(listings, checks, strict=False) if ok is True]
        dropped = len(listings) - len(live)
        if dropped:
            logger.info("career_url_validation_dropped count=%d", dropped)
        return live
    except Exception:
        logger.warning("career_url_validation_failed", exc_info=True)
        return listings  # Don't blank the response on unexpected error


async def _strategist(state: CareerState) -> dict[str, Any]:
    """Synthesize search results + resume profile into career advice via LLM.

    Attempts structured output (JobListingResponse) first for guaranteed format.
    Falls back to unstructured complete() if structured fails.
    """
    from llm.gateway import get_gateway

    gateway = get_gateway()
    system_prompt = build_system_prompt(resume_context=state.get("resume_context", ""))

    # Early return when no live data — do not hallucinate
    if not state["search_results"]:
        if state["is_morning_brief"]:
            msg = (
                "*Morning Job Brief*\n\n"
                "No new DFW positions found today. "
                "Job APIs (The Muse, Remotive, Adzuna) returned 0 new listings. "
                "Check back tomorrow — Adzuna refreshes daily."
            )
        else:
            msg = (
                "No live job listings found right now. "
                "The job APIs returned 0 results for DFW ML/AI roles. "
                "Try again tomorrow, or ask me to check a specific company's careers page."
            )
        return {
            "response": msg,
            "model_used": "none",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }

    if state["is_morning_brief"]:
        header = (
            f"[{state.get('new_job_count', 0)} new positions found today]\n\n"
            if state.get("new_job_count")
            else ""
        )
        body = (
            f"{header}Live DFW job market results:\n{state['search_results']}\n\n---\n"
            f"{MORNING_PROMPT}"
        )
        user_content = body if state["search_results"] else MORNING_PROMPT
        max_tokens = 4096
    else:
        user_input = f"<user_request>\n{state['query']}\n</user_request>"
        user_content = (
            f"Live search results:\n{state['search_results']}\n\n---\n{user_input}"
            if state["search_results"]
            else user_input
        )
        max_tokens = 4096

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
        eligible = [j for j in structured.jobs if j.is_dfw_eligible]
        eligible = await _validate_listings(eligible)
        structured.jobs = eligible
        logger.info(
            "career_strategist_structured jobs=%d skill_gap=%s",
            len(eligible),
            bool(structured.skill_gap),
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
        "response": result,
        "model_used": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
    }


# ── Graph construction ────────────────────────────────────────────────────────


def _build() -> Any:  # noqa: ANN401
    builder: StateGraph = StateGraph(CareerState)

    builder.add_node("job_searcher", _job_searcher)
    builder.add_node("strategist", _strategist)

    builder.add_edge(START, "job_searcher")
    builder.add_edge("job_searcher", "strategist")
    builder.add_edge("strategist", END)

    return builder.compile(name="career_expert")


career_subgraph = _build() if _LANGGRAPH_AVAILABLE else None
