"""
Career squad — job search, resume coaching, interview prep.

Delegates to ``career_subgraph`` (LangGraph) for the full pipeline:
  job_searcher → strategist

Falls back to a direct gateway call when LangGraph is unavailable.
"""

from __future__ import annotations

import time

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

from career.subgraph import (
    MORNING_PROMPT,
    build_search_queries,
    build_system_prompt,
    career_subgraph,
)

logger = get_logger(__name__, component="career")


class CareerSquad(BaseSquad):
    """Handles career tasks and generates morning job-search briefings."""

    @property
    def name(self) -> str:
        return SquadName.CAREER

    async def process(self, task: TaskRequest) -> TaskResult:
        t0 = time.monotonic()

        if career_subgraph is not None:
            return await self._process_via_subgraph(task, t0)
        return await self._process_direct(task, t0)

    async def _process_via_subgraph(self, task: TaskRequest, t0: float) -> TaskResult:
        import json

        history: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                history = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        result = await career_subgraph.ainvoke({
            "query": task.content,
            "routing": task.routing,
            "history": history,
            "is_morning_brief": False,
            "search_results": "",
            "response": "",
            "model_used": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        })

        return TaskResult(
            task_id=task.task_id,
            squad=SquadName.CAREER,
            content=result["response"],
            model_used=result["model_used"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
            metadata={"nodes": "job_searcher,strategist"},
        )

    async def _process_direct(self, task: TaskRequest, t0: float) -> TaskResult:
        """Fallback: single LLM call when LangGraph is unavailable."""
        import json

        from llm.gateway import get_gateway
        from search.jina import search as jina_search

        gateway = get_gateway()
        search_context = ""

        if task.routing == RoutingDecision.CLOUD:
            from core.config import get_settings
            location = get_settings().career_target_location
            search_context = await jina_search(
                f"{task.content} ML Engineer AI job {location}", max_results=5,
            )

        history: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                history = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        user_input = f"<user_request>\n{task.content}\n</user_request>"
        messages = [
            {"role": "system", "content": build_system_prompt()},
            *history,
            {"role": "user", "content": (
                f"Live search results:\n{search_context}\n\n---\n{user_input}"
                if search_context else user_input
            )},
        ]

        result = await gateway.complete(
            messages=messages, max_tokens=1500, routing=task.routing, squad=self.name,
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
        if career_subgraph is not None:
            result = await career_subgraph.ainvoke({
                "query": "",
                "routing": RoutingDecision.CLOUD,
                "history": [],
                "is_morning_brief": True,
                "search_results": "",
                "response": "",
                "model_used": "",
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
            })
            return result["response"]

        # Fallback
        from llm.gateway import get_gateway
        from search.jina import search as jina_search

        gateway = get_gateway()
        job_context = await jina_search(build_search_queries()[0], max_results=5)

        result = await gateway.complete(
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": (
                    f"Live DFW job market results:\n{job_context}\n\n---\n{MORNING_PROMPT}"
                    if job_context else MORNING_PROMPT
                )},
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
                messages=[{"role": "user", "content": "ping"}], max_tokens=5,
            )
            return bool(result.get("content"))
        except Exception:
            logger.warning("career_health_check_failed", exc_info=True)
            return False
