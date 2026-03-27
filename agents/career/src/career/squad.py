"""
Career squad — real-time job search, resume coaching, interview prep.

Grounds every response with live Jina web search so job listings,
salaries, and company intel are always current.
"""

from __future__ import annotations

import time

from core.config import get_settings
from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

logger = get_logger(__name__, component="career")

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


def _build_system_prompt() -> str:
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


def _build_job_search_queries() -> list[str]:
    s = get_settings()
    location = s.career_target_location
    roles = s.career_target_roles.replace(", ", " OR ").replace(",", " OR ")
    return [
        f"{roles} {location} hiring site:linkedin.com OR site:indeed.com",
        f"machine learning engineer jobs {location} remote apply now",
    ]


_MORNING_PROMPT = """\
Based on the live job market data above, give a crisp morning career briefing (<= 150 words):
1. One high-value action to take today (specific company to reach out to,
   specific JD to apply for, etc.)
2. One ML/AI market insight from the search results.
Keep it motivating and concrete.\
"""


class CareerSquad(BaseSquad):
    """Handles career tasks and generates morning job-search briefings."""

    @property
    def name(self) -> str:
        return SquadName.CAREER

    async def process(self, task: TaskRequest) -> TaskResult:
        from llm.gateway import get_gateway
        from search.jina import search as jina_search

        gateway = get_gateway()
        t0 = time.monotonic()

        # Ground with live search — only for cloud routing (never leak PII externally)
        search_context = ""
        if task.routing == RoutingDecision.CLOUD:
            location = get_settings().career_target_location
            search_context = await jina_search(
                f"{task.content} ML Engineer AI job {location}",
                max_results=5,
            )

        prior_turns: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                import json

                prior_turns = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        user_input = f"<user_request>\n{task.content}\n</user_request>"
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            *prior_turns,
            {
                "role": "user",
                "content": (
                    f"Live search results:\n{search_context}\n\n---\n{user_input}"
                    if search_context
                    else user_input
                ),
            },
        ]

        result = await gateway.complete(
            messages=messages,
            max_tokens=1500,
            routing=task.routing,
            squad=self.name,
        )

        return TaskResult(
            task_id=task.task_id,
            squad=SquadName.CAREER,
            content=result["content"],
            model_used=result["model"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
        )

    async def morning_brief(self) -> str:
        from llm.gateway import get_gateway
        from search.jina import search as jina_search

        gateway = get_gateway()

        job_context = await jina_search(_build_job_search_queries()[0], max_results=5)

        result = await gateway.complete(
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"Live DFW job market results:\n{job_context}\n\n---\n{_MORNING_PROMPT}"
                        if job_context
                        else _MORNING_PROMPT
                    ),
                },
            ],
            max_tokens=250,
            routing=RoutingDecision.CLOUD,
            squad=self.name,
        )
        return result["content"]

    async def health_check(self) -> bool:
        try:
            from llm.gateway import get_gateway

            result = await get_gateway().complete(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(result.get("content"))
        except Exception:
            logger.warning("career_health_check_failed", exc_info=True)
            return False
