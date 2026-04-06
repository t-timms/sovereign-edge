"""
Intelligence expert — research synthesis, AI/ML trend monitoring, news digest.

Delegates to ``intelligence_subgraph`` (LangGraph) for the full pipeline:
  arxiv_fetcher + hf_fetcher (parallel) → ranker → synthesizer

Falls back to a direct gateway call when LangGraph is unavailable.
"""

from __future__ import annotations

import asyncio
import time

from core.expert import BaseExpert
from core.types import ExpertName, RoutingDecision, TaskRequest, TaskResult
from observability.logging import get_logger

from intelligence.subgraph import (
    MORNING_PROMPT,
    SYSTEM_PROMPT,
    BriefOutput,
    intelligence_subgraph,
)

logger = get_logger(__name__, component="intelligence")


class IntelligenceExpert(BaseExpert):
    """Research synthesis and trend monitoring grounded with live papers."""

    @property
    def name(self) -> str:
        return ExpertName.INTELLIGENCE

    async def process(self, task: TaskRequest) -> TaskResult:
        t0 = time.monotonic()

        if intelligence_subgraph is not None:
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

        try:
            result = await intelligence_subgraph.ainvoke(
                {
                    "query": task.content,
                    "routing": task.routing,
                    "history": history,
                    "is_morning_brief": False,
                    "raw_papers": [],
                    "ranked_papers": [],
                    "repo_relevant_papers": [],
                    "response": "",
                    "model_used": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                }
            )
        except Exception:
            logger.warning("intel_subgraph_invoke_failed — falling back to direct", exc_info=True)
            return await self._process_direct(task, t0)

        return TaskResult(
            task_id=task.task_id,
            expert=ExpertName.INTELLIGENCE,
            content=result["response"],
            model_used=result["model_used"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
            metadata={"nodes": "arxiv_fetcher,hf_fetcher,ranker,synthesizer"},
        )

    async def _process_direct(self, task: TaskRequest, t0: float) -> TaskResult:
        """Fallback: single LLM call when LangGraph is unavailable."""
        import json

        from llm.gateway import get_gateway
        from search.arxiv import fetch_recent, format_papers
        from search.hf import fetch_daily_papers, format_hf_papers

        gateway = get_gateway()
        research_context = ""

        if task.routing == RoutingDecision.CLOUD:
            arxiv_papers, hf_papers = await asyncio.gather(
                fetch_recent(max_results=5),
                fetch_daily_papers(),
                return_exceptions=True,
            )
            if isinstance(arxiv_papers, Exception):
                logger.warning("arxiv_fetch_failed", error=str(arxiv_papers))
                arxiv_papers = []
            if isinstance(hf_papers, Exception):
                logger.warning("hf_fetch_failed", error=str(hf_papers))
                hf_papers = []
            all_parts: list[str] = []
            if arxiv_papers:
                all_parts.append(format_papers(arxiv_papers))
            if hf_papers:
                all_parts.append(format_hf_papers(hf_papers))
            research_context = "Recent AI/ML papers:\n" + "\n".join(all_parts) if all_parts else ""

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
                    f"Live research data:\n{research_context}\n\n---\n{user_input}"
                    if research_context
                    else user_input
                ),
            },
        ]

        result = await gateway.complete(
            messages=messages,
            max_tokens=2048,
            routing=task.routing,
            expert=self.name,
        )

        brief = BriefOutput(content=result["content"])
        if not brief.is_valid:
            logger.warning(
                "brief_quality_low expert=intelligence links=%d words=%d",
                brief.link_count,
                brief.word_count,
            )

        return TaskResult(
            task_id=task.task_id,
            expert=ExpertName.INTELLIGENCE,
            content=brief.content,
            model_used=result["model"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
        )

    async def morning_brief(self) -> str:
        if intelligence_subgraph is not None:
            try:
                result = await intelligence_subgraph.ainvoke(
                    {
                        "query": "",
                        "routing": RoutingDecision.CLOUD,
                        "history": [],
                        "is_morning_brief": True,
                        "raw_papers": [],
                        "ranked_papers": [],
                        "repo_relevant_papers": [],
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
                    "intel_morning_brief_subgraph_failed — falling back to direct", exc_info=True
                )
                # fall through to direct path below

        # Fallback
        from llm.gateway import get_gateway
        from search.arxiv import fetch_recent, format_papers
        from search.hf import fetch_daily_papers, format_hf_papers

        gateway = get_gateway()
        arxiv_papers, hf_papers = await asyncio.gather(
            fetch_recent(max_results=5),
            fetch_daily_papers(),
            return_exceptions=True,
        )
        if isinstance(arxiv_papers, Exception):
            logger.warning("morning_brief_arxiv_failed", error=str(arxiv_papers))
            arxiv_papers = []
        if isinstance(hf_papers, Exception):
            logger.warning("morning_brief_hf_failed", error=str(hf_papers))
            hf_papers = []

        all_parts: list[str] = []
        if arxiv_papers:
            all_parts.append(format_papers(arxiv_papers))
        if hf_papers:
            all_parts.append(format_hf_papers(hf_papers))
        research_context = "Recent AI/ML papers:\n" + "\n".join(all_parts) if all_parts else ""

        result = await gateway.complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Live research data:\n{research_context}\n\n---\n{MORNING_PROMPT}"
                        if research_context
                        else MORNING_PROMPT
                    ),
                },
            ],
            max_tokens=300,
            routing=RoutingDecision.CLOUD,
            expert=self.name,
        )
        brief = BriefOutput(content=result["content"])
        if not brief.is_valid:
            logger.warning(
                "morning_brief_quality_low links=%d words=%d",
                brief.link_count,
                brief.word_count,
            )
        return brief.content

    async def stream_process(self, task: TaskRequest):  # type: ignore[override]  # noqa: ANN201
        """Token-by-token streaming — runs direct gateway call (subgraph streaming WIP)."""
        import json

        from llm.gateway import get_gateway
        from search.arxiv import fetch_recent, format_papers
        from search.hf import fetch_daily_papers, format_hf_papers

        gateway = get_gateway()
        research_context = ""

        if task.routing == RoutingDecision.CLOUD:
            arxiv_papers, hf_papers = await asyncio.gather(
                fetch_recent(max_results=5),
                fetch_daily_papers(),
                return_exceptions=True,
            )
            if isinstance(arxiv_papers, Exception):
                logger.warning("stream_arxiv_failed", error=str(arxiv_papers))
                arxiv_papers = []
            if isinstance(hf_papers, Exception):
                logger.warning("stream_hf_failed", error=str(hf_papers))
                hf_papers = []
            all_parts: list[str] = []
            if arxiv_papers:
                all_parts.append(format_papers(arxiv_papers))
            if hf_papers:
                all_parts.append(format_hf_papers(hf_papers))
            research_context = "Recent AI/ML papers:\n" + "\n".join(all_parts) if all_parts else ""

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
                    f"Live research data:\n{research_context}\n\n---\n{user_input}"
                    if research_context
                    else user_input
                ),
            },
        ]

        async for chunk in gateway.stream_complete(
            messages=messages,
            max_tokens=2048,
            routing=task.routing,
            expert=self.name,
        ):
            yield chunk

    async def health_check(self) -> bool:
        try:
            from llm.gateway import get_gateway

            result = await get_gateway().complete(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(result.get("content"))
        except Exception:
            logger.warning("intelligence_health_check_failed", exc_info=True)
            return False
