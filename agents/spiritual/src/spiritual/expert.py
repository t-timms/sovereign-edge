"""
Spiritual expert — faith formation, prayer, scripture study, devotionals.

Delegates to ``spiritual_subgraph`` (LangGraph) for the full pipeline:
  scripture_fetcher → theologian

Falls back to a direct gateway call when LangGraph is unavailable.
"""

from __future__ import annotations

import json
import time

from core.expert import BaseExpert
from core.types import ExpertName, RoutingDecision, TaskRequest, TaskResult
from observability.logging import get_logger

from spiritual.subgraph import (
    DEVOTIONAL_PROMPT,
    SYSTEM_PROMPT,
    spiritual_subgraph,
)

logger = get_logger(__name__, component="spiritual")


class SpiritualExpert(BaseExpert):
    """Handles faith-formation tasks and generates morning devotionals."""

    @property
    def name(self) -> str:
        return ExpertName.SPIRITUAL

    async def process(self, task: TaskRequest) -> TaskResult:
        t0 = time.monotonic()

        if spiritual_subgraph is not None:
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
            result = await spiritual_subgraph.ainvoke(
                {
                    "query": task.content,
                    "routing": task.routing,
                    "history": history,
                    "is_morning_brief": False,
                    "scripture": "",
                    "response": "",
                    "model_used": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                }
            )
        except Exception:
            logger.warning(
                "spiritual_subgraph_invoke_failed — falling back to direct", exc_info=True
            )
            return await self._process_direct(task, t0)

        return TaskResult(
            task_id=task.task_id,
            expert=ExpertName.SPIRITUAL,
            content=result["response"],
            model_used=result["model_used"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
            metadata={"nodes": "scripture_fetcher,theologian"},
        )

    async def _process_direct(self, task: TaskRequest, t0: float) -> TaskResult:
        """Fallback: single LLM call when LangGraph is unavailable."""

        from llm.gateway import get_gateway
        from search.bible import extract_reference, format_verse, lookup, random_verse

        gateway = get_gateway()
        scripture_context = ""

        if task.routing == RoutingDecision.CLOUD:
            ref = extract_reference(task.content)
            verse = await lookup(ref) if ref else await random_verse()
            formatted = format_verse(verse)
            if formatted:
                scripture_context = f"Scripture:\n{formatted}"

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
                    f"{scripture_context}\n\n---\n{user_input}" if scripture_context else user_input
                ),
            },
        ]

        content = await gateway.complete(
            messages=messages,
            max_tokens=1024,
            routing=task.routing,
            expert=self.name,
        )

        return TaskResult(
            task_id=task.task_id,
            expert=ExpertName.SPIRITUAL,
            content=content,
            model_used="",
            tokens_in=0,
            tokens_out=0,
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=0.0,
            routing=task.routing,
        )

    async def morning_brief(self) -> str:
        if spiritual_subgraph is not None:
            try:
                result = await spiritual_subgraph.ainvoke(
                    {
                        "query": "",
                        "routing": RoutingDecision.CLOUD,
                        "history": [],
                        "is_morning_brief": True,
                        "scripture": "",
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
                    "spiritual_subgraph_morning_brief_failed — using direct", exc_info=True
                )

        # Fallback
        from llm.gateway import get_gateway
        from search.bible import format_verse, random_verse

        gateway = get_gateway()
        verse = await random_verse()
        verse_text = format_verse(verse)

        return await gateway.complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Scripture:\n{verse_text}\n\n---\n{DEVOTIONAL_PROMPT}"
                        if verse_text
                        else (
                            "Generate a brief morning devotional with a scripture verse, "
                            "2-3 sentences of reflection, and a one-sentence prayer. Under 120 words."  # noqa: E501
                        )
                    ),
                },
            ],
            max_tokens=300,
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
            logger.warning("spiritual_health_check_failed", exc_info=True)
            return False
