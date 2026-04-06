"""
Creative expert — writing, content strategy, social media, storytelling.

Delegates to ``creative_subgraph`` (LangGraph) for the full pipeline:
  trend_researcher → writer

Falls back to a direct gateway call when LangGraph is unavailable.
"""

from __future__ import annotations

import json
import time

from core.expert import BaseExpert
from core.types import ExpertName, RoutingDecision, TaskRequest, TaskResult
from observability.logging import get_logger

from creative.subgraph import (
    MORNING_PROMPT,
    SYSTEM_PROMPT,
    creative_subgraph,
)

logger = get_logger(__name__, component="creative")


class CreativeExpert(BaseExpert):
    """Handles creative writing tasks and generates daily creative prompts."""

    @property
    def name(self) -> str:
        return ExpertName.CREATIVE

    async def process(self, task: TaskRequest) -> TaskResult:
        t0 = time.monotonic()

        if creative_subgraph is not None:
            return await self._process_via_subgraph(task, t0)
        return await self._process_direct(task, t0)

    async def _process_via_subgraph(self, task: TaskRequest, t0: float) -> TaskResult:

        history: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                history = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        try:
            result = await creative_subgraph.ainvoke(
                {
                    "query": task.content,
                    "routing": task.routing,
                    "history": history,
                    "is_morning_brief": False,
                    "trend_context": "",
                    "response": "",
                    "model_used": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                }
            )
        except Exception:
            logger.warning(
                "creative_subgraph_invoke_failed — falling back to direct", exc_info=True
            )  # noqa: E501
            return await self._process_direct(task, t0)

        return TaskResult(
            task_id=task.task_id,
            expert=ExpertName.CREATIVE,
            content=result["response"],
            model_used=result["model_used"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
            metadata={"nodes": "trend_researcher,writer"},
        )

    async def _process_direct(self, task: TaskRequest, t0: float) -> TaskResult:
        """Fallback: single LLM call when LangGraph is unavailable."""

        from llm.gateway import get_gateway
        from search.jina import search as jina_search

        gateway = get_gateway()
        trend_context = ""

        if task.routing == RoutingDecision.CLOUD:
            trend_context = await jina_search(
                f"{task.content} content strategy examples 2026",
                max_results=3,
            )

        history: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                history = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        user_input = f"<user_request>\n{task.content}\n</user_request>"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": (
                    f"Current trends and context:\n{trend_context}\n\n---\n{user_input}"
                    if trend_context
                    else user_input
                ),
            },
        ]

        content = await gateway.complete(
            messages=messages,
            max_tokens=2048,
            routing=task.routing,
            expert=self.name,
        )

        return TaskResult(
            task_id=task.task_id,
            expert=ExpertName.CREATIVE,
            content=content,
            model_used="",
            tokens_in=0,
            tokens_out=0,
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=0.0,
            routing=task.routing,
        )

    async def morning_brief(self) -> str:
        if creative_subgraph is not None:
            try:
                result = await creative_subgraph.ainvoke(
                    {
                        "query": "",
                        "routing": RoutingDecision.CLOUD,
                        "history": [],
                        "is_morning_brief": True,
                        "trend_context": "",
                        "response": "",
                        "model_used": "",
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "cost_usd": 0.0,
                    }
                )
                return result["response"]
            except Exception:
                logger.warning(
                    "creative_subgraph_morning_brief_failed — using direct", exc_info=True
                )  # noqa: E501

        # Fallback
        from llm.gateway import get_gateway
        from search.jina import search as jina_search

        gateway = get_gateway()
        trend_context = await jina_search(
            "AI content creation trends 2026 creator economy",
            max_results=3,
        )

        return await gateway.complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Current trends and context:\n{trend_context}\n\n---\n{MORNING_PROMPT}"
                        if trend_context
                        else MORNING_PROMPT
                    ),
                },
            ],
            max_tokens=150,
            routing=RoutingDecision.CLOUD,
            expert=self.name,
        )

    async def health_check(self) -> bool:
        try:
            from llm.gateway import get_gateway

            result = await get_gateway().complete(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(result)
        except Exception:
            logger.warning("creative_health_check_failed", exc_info=True)
            return False
