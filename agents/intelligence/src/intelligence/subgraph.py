"""
Intelligence expert — LangGraph subgraph.

Pipeline:
    START
      ├─► arxiv_fetcher  (parallel, same superstep)
      └─► hf_fetcher
              │
              ▼ (both must finish)
           ranker          — keyword-based relevance scorer; no LLM call
              │
              ▼
         synthesizer       — LLM generates the final brief
              │
             END

The subgraph exposes a compiled ``intelligence_subgraph`` instance that
the expert and the director can invoke directly.  Falls back to ``None``
when LangGraph is not installed.
"""

from __future__ import annotations

import logging
import operator
import re
from typing import Annotated, Any

from pydantic import BaseModel, model_validator

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

# ── Relevance keywords for the ranker node ────────────────────────────────────

_RELEVANCE_KEYWORDS: frozenset[str] = frozenset({
    "fine-tuning", "lora", "qlora", "grpo", "rlhf", "dpo", "orpo",
    "inference", "vllm", "tensorrt", "quantization", "flash attention",
    "agent", "langgraph", "mcp", "tool use", "reasoning",
    "llm", "transformer", "diffusion", "multimodal", "benchmark",
})

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
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

MORNING_PROMPT = """\
Based on the live research data above, generate a concise intelligence briefing
(≤ 200 words):
1. One significant AI/ML development from the papers above worth knowing today.
2. One technique or finding directly relevant to LLM fine-tuning or inference optimization.
3. One actionable insight — something to try or watch for.
Be specific. Cite paper titles and link them.\
"""


# ── Output validator ──────────────────────────────────────────────────────────

class BriefOutput(BaseModel):
    """Validates structural quality of an intelligence brief before delivery."""

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


# ── State ─────────────────────────────────────────────────────────────────────

class IntelligenceState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────
    query: str
    routing: str                              # RoutingDecision value
    history: list[dict[str, str]]             # prior conversation turns
    is_morning_brief: bool
    # ── Intermediate ──────────────────────────────────────────────────────
    # operator.add merges parallel writes from arxiv_fetcher + hf_fetcher
    raw_papers: Annotated[list[dict], operator.add]
    ranked_papers: list[dict]
    # ── Outputs ───────────────────────────────────────────────────────────
    response: str
    model_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def _arxiv_fetcher(state: IntelligenceState) -> dict[str, Any]:
    """Fetch recent AI/ML papers from arXiv (cloud-only)."""
    from core.types import RoutingDecision

    if state["routing"] != RoutingDecision.CLOUD:
        return {"raw_papers": []}
    try:
        from search.arxiv import fetch_recent

        papers = await fetch_recent(max_results=5)
        logger.info("intel_arxiv_fetched count=%d", len(papers))
        return {"raw_papers": papers}
    except Exception:
        logger.warning("intel_arxiv_fetch_failed", exc_info=True)
        return {"raw_papers": []}


async def _hf_fetcher(state: IntelligenceState) -> dict[str, Any]:
    """Fetch trending papers from HuggingFace Daily Papers (cloud-only)."""
    from core.types import RoutingDecision

    if state["routing"] != RoutingDecision.CLOUD:
        return {"raw_papers": []}
    try:
        from search.hf import fetch_daily_papers

        papers = await fetch_daily_papers()
        logger.info("intel_hf_fetched count=%d", len(papers))
        return {"raw_papers": papers}
    except Exception:
        logger.warning("intel_hf_fetch_failed", exc_info=True)
        return {"raw_papers": []}


# Lazy-loaded FlashRank cross-encoder (~4 MB model, CPU-only, downloaded on first use)
_flashrank_ranker: Any = None


def _get_flashrank() -> Any:
    global _flashrank_ranker
    if _flashrank_ranker is None:
        try:
            from flashrank import Ranker

            _flashrank_ranker = Ranker(
                model_name="ms-marco-MiniLM-L-4-v2", cache_dir="/tmp/flashrank"
            )
            logger.info("intel_flashrank_loaded")
        except ImportError:
            _flashrank_ranker = False
            logger.debug("intel_flashrank_unavailable — using keyword fallback")
    return _flashrank_ranker if _flashrank_ranker else None


def _ranker(state: IntelligenceState) -> dict[str, Any]:
    """Score papers by semantic relevance; return top 8. No LLM call.

    Uses FlashRank cross-encoder when available (recommended). Falls back to
    keyword counting when flashrank is not installed or the query is empty.
    Install: uv add flashrank  (adds ~4 MB ONNX model on first run).
    """
    papers = state["raw_papers"]
    if not papers:
        return {"ranked_papers": []}

    ranker = _get_flashrank()
    if ranker is not None and state.get("query"):
        try:
            from flashrank import RerankRequest

            passages = [
                {
                    "id": i,
                    "text": (p.get("title", "") + " " + p.get("summary", "")).strip(),
                }
                for i, p in enumerate(papers)
            ]
            req = RerankRequest(query=state["query"], passages=passages)
            results = ranker.rerank(req)
            top_ids = [r["id"] for r in results[:8]]
            ranked = [papers[i] for i in top_ids]
            logger.info("intel_ranker_flashrank input=%d top=%d", len(papers), len(ranked))
            return {"ranked_papers": ranked}
        except Exception:  # graceful fallback to keyword scorer
            logger.warning("intel_ranker_flashrank_failed — falling back to keyword", exc_info=True)

    # Keyword fallback
    def _score(paper: dict) -> int:
        text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
        return sum(1 for kw in _RELEVANCE_KEYWORDS if kw in text)

    ranked_kw = sorted(papers, key=_score, reverse=True)
    top = ranked_kw[:8]
    logger.info("intel_ranker_keyword input=%d top=%d", len(papers), len(top))
    return {"ranked_papers": top}


async def _synthesizer(state: IntelligenceState) -> dict[str, Any]:
    """Generate the intelligence brief from ranked papers via LLM."""
    from llm.gateway import get_gateway

    gateway = get_gateway()

    # Build paper context string
    research_context = ""
    if state["ranked_papers"]:
        try:
            from search.arxiv import format_papers

            research_context = "Recent AI/ML papers:\n" + format_papers(
                state["ranked_papers"]
            )
        except Exception:
            logger.warning("intel_format_papers_failed", exc_info=True)

    if state["is_morning_brief"]:
        user_content = (
            f"Live research data:\n{research_context}\n\n---\n{MORNING_PROMPT}"
            if research_context
            else MORNING_PROMPT
        )
        max_tokens = 300
    else:
        user_input = f"<user_request>\n{state['query']}\n</user_request>"
        user_content = (
            f"Live research data:\n{research_context}\n\n---\n{user_input}"
            if research_context
            else user_input
        )
        max_tokens = 2048

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state["history"],
        {"role": "user", "content": user_content},
    ]

    result = await gateway.complete(
        messages=messages,
        max_tokens=max_tokens,
        routing=state["routing"],
        expert="intelligence",
    )

    brief = BriefOutput(content=result["content"])
    if not brief.is_valid:
        logger.warning(
            "intel_brief_quality_low links=%d words=%d",
            brief.link_count,
            brief.word_count,
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
    builder: StateGraph = StateGraph(IntelligenceState)

    builder.add_node("arxiv_fetcher", _arxiv_fetcher)
    builder.add_node("hf_fetcher", _hf_fetcher)
    builder.add_node("ranker", _ranker)
    builder.add_node("synthesizer", _synthesizer)

    # Parallel fetch — both nodes start in the same superstep
    builder.add_edge(START, "arxiv_fetcher")
    builder.add_edge(START, "hf_fetcher")

    # ranker waits for both fetchers to complete
    builder.add_edge("arxiv_fetcher", "ranker")
    builder.add_edge("hf_fetcher", "ranker")

    builder.add_edge("ranker", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(name="intelligence_expert")


intelligence_subgraph = _build() if _LANGGRAPH_AVAILABLE else None
