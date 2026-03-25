"""
Career squad — real-time job search, resume coaching, interview prep.

Grounds every response with live Jina web search so job listings,
salaries, and company intel are always current.
"""

from __future__ import annotations

import time

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

logger = get_logger(__name__, component="career")

_SYSTEM_PROMPT = """\
You are the Career Intelligence of Sovereign Edge — a world-class career strategist
specializing in ML Engineering, AI Engineering, and LLM Engineering roles in the
Dallas-Fort Worth metro.

You have access to live search results. When job listings are provided, extract and
present: company name, role title, location, salary range (if shown), and a direct
application link. Format as a clean table or numbered list.

Emphasize the user's differentiators: GRPO fine-tuning, LangGraph agents, MCP server
development, vLLM/TensorRT-LLM production serving, structured outputs, LLMOps,
Blackwell GPU (RTX 5070 Ti) hands-on experience.

Be direct, specific, and actionable. When no search results are available, draw on
deep knowledge of the DFW ML market.\
"""

_MORNING_PROMPT = """\
Based on the live job market data above, give a crisp morning career briefing (<= 150 words):
1. One high-value action to take today (specific company to reach out to,
   specific JD to apply for, etc.)
2. One DFW ML/AI market insight from the search results.
Keep it motivating and concrete.\
"""

# Search queries for DFW ML/AI job market
_JOB_SEARCH_QUERIES = [
    (
        "ML Engineer LLM Engineer AI Engineer Dallas Fort Worth Texas hiring 2026"
        " site:linkedin.com OR site:indeed.com"
    ),
    "machine learning engineer jobs DFW Texas remote apply now",
]


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
            search_context = await jina_search(
                f"{task.content} ML Engineer AI job Dallas Fort Worth Texas 2026",
                max_results=5,
            )

        prior_turns: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                import json

                prior_turns = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *prior_turns,
            {
                "role": "user",
                "content": (
                    f"Live search results:\n{search_context}\n\n---\n{task.content}"
                    if search_context
                    else task.content
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

        job_context = await jina_search(_JOB_SEARCH_QUERIES[0], max_results=5)

        result = await gateway.complete(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
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
