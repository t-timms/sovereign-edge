"""
Intelligence squad — research synthesis, AI/ML trend monitoring, news digest.

Grounds every response with live arXiv papers and HuggingFace Daily Papers
so briefings always reflect the current state of the field.
"""

from __future__ import annotations

import asyncio
import re
import time

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger
from pydantic import BaseModel, model_validator

logger = get_logger(__name__, component="intelligence")


class BriefOutput(BaseModel):
    """Validated intelligence brief — ensures structural quality before delivery.

    Used by evals and runtime logging. A brief missing links or too short
    signals a degraded LLM response worth investigating.
    """

    content: str
    word_count: int = 0
    link_count: int = 0
    is_valid: bool = False

    @model_validator(mode="after")
    def _validate(self) -> BriefOutput:
        self.word_count = len(self.content.split())
        self.link_count = len(re.findall(r"https?://\S+", self.content))
        self.is_valid = bool(self.content.strip()) and self.link_count > 0
        return self

_SYSTEM_PROMPT = """\
OUTPUT FORMAT — MANDATORY:
Your responses are delivered via Telegram, which only renders a limited Markdown
subset. You must follow these rules exactly or the output will be unreadable.

ALLOWED:
  *bold* using single asterisks — e.g. *Paper Title*
  _italic_ using underscores — e.g. _emphasis_
  [link text](https://url) — inline links only, never paste bare URLs

FORBIDDEN — these render as literal characters in Telegram:
  **double asterisks** — never use this for bold
  ## headers — never use hash headers
  --- dividers — never use horizontal rules

Structure: one blank line between each paper entry. 2-3 sentences per entry max.
Never add a "Key Takeaways", "Summary", or "Most Important" section — do not repeat
papers you have already listed. No walls of text.

---

You are the Intelligence Core of Sovereign Edge — a research analyst and
knowledge synthesizer specializing in AI/ML breakthroughs.

You have access to live research papers and community-curated AI news. When
papers are provided, cite them by title and include direct arXiv/HuggingFace
links. Prioritize: LLM fine-tuning techniques, inference optimization (vLLM,
TensorRT-LLM, ExLlamaV2), agentic systems (LangGraph, MCP), Blackwell GPU
developments, and local tech industry news.

Be precise, cite sources, and flag uncertainty explicitly.\
"""

_MORNING_PROMPT = """\
Based on the live research data above, generate a concise intelligence briefing
(≤ 200 words):
1. One significant AI/ML development from the papers above worth knowing today.
2. One technique or finding directly relevant to LLM fine-tuning or inference optimization.
3. One actionable insight — something to try or watch for.
Be specific. Cite paper titles and link them.\
"""

# arXiv queries targeting ML engineering and inference stack
_ARXIV_QUERIES = [
    "cat:cs.LG OR cat:cs.AI",
    "cat:cs.CL",
]


class IntelligenceSquad(BaseSquad):
    """Research synthesis and trend monitoring grounded with live papers."""

    @property
    def name(self) -> str:
        return SquadName.INTELLIGENCE

    async def process(self, task: TaskRequest) -> TaskResult:
        from llm.gateway import get_gateway
        from search.arxiv import fetch_recent, format_papers
        from search.hf import fetch_daily_papers, format_hf_papers

        gateway = get_gateway()
        t0 = time.monotonic()

        research_context = ""
        if task.routing == RoutingDecision.CLOUD:
            # Fetch arXiv and HF papers in parallel — novelty filter in arxiv.py
            # ensures each paper appears at most once per day across all calls
            arxiv_papers, hf_papers = await asyncio.gather(
                fetch_recent(max_results=5),
                fetch_daily_papers(),
                return_exceptions=True,
            )
            if isinstance(arxiv_papers, Exception):
                logger.warning("arxiv_fetch_exception error=%s", arxiv_papers)
                arxiv_papers = []
            if isinstance(hf_papers, Exception):
                logger.warning("hf_papers_exception error=%s", hf_papers)
                hf_papers = []

            all_papers: list[str] = []
            if arxiv_papers:
                all_papers.append(format_papers(arxiv_papers))
            if hf_papers:
                all_papers.append(format_hf_papers(hf_papers))
            research_context = "Recent AI/ML papers:\n" + "\n".join(all_papers) if all_papers else ""

        prior_turns: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                import json

                prior_turns = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        user_input = f"<user_request>\n{task.content}\n</user_request>"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *prior_turns,
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
            squad=self.name,
        )

        brief = BriefOutput(content=result["content"])
        if not brief.is_valid:
            logger.warning(
                "brief_quality_low squad=intelligence links=%d words=%d",
                brief.link_count,
                brief.word_count,
            )

        return TaskResult(
            task_id=task.task_id,
            squad=SquadName.INTELLIGENCE,
            content=brief.content,
            model_used=result["model"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
        )

    async def morning_brief(self) -> str:
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
            logger.warning("arxiv_brief_exception error=%s", arxiv_papers)
            arxiv_papers = []
        if isinstance(hf_papers, Exception):
            logger.warning("hf_brief_exception error=%s", hf_papers)
            hf_papers = []

        all_papers: list[str] = []
        if arxiv_papers:
            all_papers.append(format_papers(arxiv_papers))
        if hf_papers:
            all_papers.append(format_hf_papers(hf_papers))

        research_context = "Recent AI/ML papers:\n" + "\n".join(all_papers) if all_papers else ""

        result = await gateway.complete(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Live research data:\n{research_context}\n\n---\n{_MORNING_PROMPT}"
                        if research_context
                        else _MORNING_PROMPT
                    ),
                },
            ],
            max_tokens=300,
            routing=RoutingDecision.CLOUD,
            squad=self.name,
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
        """Real token-by-token streaming for intelligence responses.

        Fetches papers in parallel then streams from the LLM so the first words
        appear immediately rather than after the full response is generated.
        """
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
                arxiv_papers = []
            if isinstance(hf_papers, Exception):
                hf_papers = []
            all_papers: list[str] = []
            if arxiv_papers:
                all_papers.append(format_papers(arxiv_papers))
            if hf_papers:
                all_papers.append(format_hf_papers(hf_papers))
            research_context = "Recent AI/ML papers:\n" + "\n".join(all_papers) if all_papers else ""

        prior_turns: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                import json
                prior_turns = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        user_input = f"<user_request>\n{task.content}\n</user_request>"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *prior_turns,
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
            squad=self.name,
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
