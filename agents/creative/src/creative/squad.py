"""
Creative squad — writing, content strategy, social media, storytelling.

Grounds creative work with live Jina search so content references current
trends, real examples, and up-to-date platform conventions.
"""

from __future__ import annotations

import time

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

logger = get_logger(__name__, component="creative")

_SYSTEM_PROMPT = """\
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

_MORNING_PROMPT = """\
Based on the current trends and examples above, generate one creative
micro-challenge for today (≤ 100 words). Choose from:
- A writing exercise tied to a current trend or format
- A content angle that is fresh right now in the creator space
- A storytelling technique worth practicing

Make it specific, immediately actionable, and completable in 15-20 minutes.\
"""

_TREND_QUERIES = [
    "AI content creation trends 2026 creator economy",
    "LinkedIn content strategy technical professionals 2026",
]


class CreativeSquad(BaseSquad):
    """Handles creative writing tasks and generates daily creative prompts."""

    @property
    def name(self) -> str:
        return SquadName.CREATIVE

    async def process(self, task: TaskRequest) -> TaskResult:
        from llm.gateway import get_gateway
        from search.jina import search as jina_search

        gateway = get_gateway()
        t0 = time.monotonic()

        trend_context = ""
        if task.routing == RoutingDecision.CLOUD:
            # Search for context relevant to the creative task
            trend_context = await jina_search(
                f"{task.content} content strategy examples 2026",
                max_results=3,
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
                    f"Current trends and context:\n{trend_context}\n\n---\n{task.content}"
                    if trend_context
                    else task.content
                ),
            },
        ]

        result = await gateway.complete(
            messages=messages,
            max_tokens=2048,
            routing=task.routing,
            squad=self.name,
        )

        return TaskResult(
            task_id=task.task_id,
            squad=SquadName.CREATIVE,
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

        trend_context = await jina_search(_TREND_QUERIES[0], max_results=3)

        result = await gateway.complete(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Current trends and context:\n{trend_context}\n\n---\n{_MORNING_PROMPT}"
                        if trend_context
                        else _MORNING_PROMPT
                    ),
                },
            ],
            max_tokens=150,
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
            logger.warning("creative_health_check_failed", exc_info=True)
            return False
