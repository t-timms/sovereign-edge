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

from pydantic import BaseModel, Field, model_validator

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

_RELEVANCE_KEYWORDS: frozenset[str] = frozenset(
    {
        "fine-tuning",
        "lora",
        "qlora",
        "grpo",
        "rlhf",
        "dpo",
        "orpo",
        "inference",
        "vllm",
        "tensorrt",
        "quantization",
        "flash attention",
        "agent",
        "langgraph",
        "mcp",
        "tool use",
        "reasoning",
        "llm",
        "transformer",
        "diffusion",
        "multimodal",
        "benchmark",
    }
)

# ── Prompts ───────────────────────────────────────────────────────────────────
# Loaded from versioned YAML (prompts/intelligence/v*.yaml) with inline fallback.

_FALLBACK_SYSTEM = """\
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

_FALLBACK_MORNING = """\
Based on the live research data above, generate a concise intelligence briefing
(≤ 250 words):
1. One significant AI/ML development from the papers above worth knowing today.
2. One technique or finding directly relevant to LLM fine-tuning or inference optimization.
3. One actionable insight — something to try or watch for.
4. PORTFOLIO GAP CHECK: If any paper introduces a technique with no matching repo listed
   above, add one sentence: "*New project idea:* [paper technique] → [proposed project
   name] — [1-sentence resume impact for ML Engineer job search]."
   Omit section 4 entirely if all papers already match an existing repo.
Be specific. Cite paper titles and link them.\
"""

try:
    from core.prompts import get_prompt_field, get_system_prompt

    SYSTEM_PROMPT = get_system_prompt("intelligence") or _FALLBACK_SYSTEM
    MORNING_PROMPT = get_prompt_field("intelligence", "morning_prompt") or _FALLBACK_MORNING
except Exception:
    logger.debug("prompt_loader_unavailable — using inline fallback")
    SYSTEM_PROMPT = _FALLBACK_SYSTEM
    MORNING_PROMPT = _FALLBACK_MORNING


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


# ── Structured output models ──────────────────────────────────────────────────


class PaperEntry(BaseModel):
    """One paper in the intelligence brief."""

    title: str = Field(description="Full paper title")
    url: str = Field(description="Direct arXiv or HuggingFace link")
    key_finding: str = Field(description="1-2 sentence summary of the key result or technique")
    repos_matched: list[str] = Field(
        default_factory=list,
        description="Names of local repos this paper is relevant to, if any",
    )


class IntelBriefResponse(BaseModel):
    """Structured intelligence brief — 3-5 papers + highlights."""

    papers: list[PaperEntry] = Field(description="Top 3-5 papers from the research data")
    technique_highlight: str = Field(
        default="",
        description="One fine-tuning or inference technique worth knowing today (1 sentence)",
    )
    actionable_insight: str = Field(
        default="",
        description="One concrete thing to try or watch for this week (1 sentence)",
    )
    project_suggestion: str = Field(
        default="",
        description=(
            "If any paper introduces a technique with no matching repo, suggest one new "
            "portfolio project in one sentence: 'Build X — strengthens Y for ML job search'. "
            "Empty string if all papers already match an existing repo."
        ),
    )


def format_intel_brief(brief: IntelBriefResponse, repo_relevant: list[dict]) -> str:
    """Render an IntelBriefResponse as Telegram-safe numbered markdown."""
    # Build a fast lookup: title → repos from the ranker's annotations
    ranker_repos: dict[str, list[str]] = {
        p.get("title", ""): p.get("repos", []) for p in repo_relevant
    }

    lines: list[str] = []
    for i, paper in enumerate(brief.papers, 1):
        # Prefer repos from ranker annotation; fall back to LLM-supplied
        repos = ranker_repos.get(paper.title) or paper.repos_matched
        repo_tag = f" → _{', '.join(repos)}_" if repos else ""
        lines.append(f"{i}. *{paper.title}*{repo_tag}")
        lines.append(paper.key_finding)
        lines.append(f"[Read →]({paper.url})")
        lines.append("")

    if brief.technique_highlight:
        lines.append(f"*Technique:* {brief.technique_highlight}")
    if brief.actionable_insight:
        lines.append(f"*Try this:* {brief.actionable_insight}")
    if brief.project_suggestion:
        lines.append("")
        lines.append(f"*New project idea:* {brief.project_suggestion}")

    return "\n".join(lines).strip()


# ── State ─────────────────────────────────────────────────────────────────────


class IntelligenceState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────
    query: str
    routing: str  # RoutingDecision value
    history: list[dict[str, str]]  # prior conversation turns
    is_morning_brief: bool
    # ── Intermediate ──────────────────────────────────────────────────────
    # operator.add merges parallel writes from arxiv_fetcher + hf_fetcher
    raw_papers: Annotated[list[dict], operator.add]
    ranked_papers: list[dict]
    # papers that match at least one local repo — subset of ranked_papers
    repo_relevant_papers: list[dict]
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


def _get_flashrank() -> Any:  # noqa: ANN401
    global _flashrank_ranker
    if _flashrank_ranker is None:
        try:
            import tempfile
            from pathlib import Path

            from flashrank import Ranker

            _flashrank_ranker = Ranker(
                model_name="ms-marco-TinyBERT-L-2-v2",
                cache_dir=str(Path(tempfile.gettempdir()) / "flashrank"),
            )
            logger.info("intel_flashrank_loaded")
        except ImportError:
            _flashrank_ranker = False
            logger.debug("intel_flashrank_unavailable — using keyword fallback")
        except Exception:
            # Catches model download failures, ONNX runtime errors, temp-dir permission issues
            _flashrank_ranker = False
            logger.warning("intel_flashrank_init_failed — using keyword fallback", exc_info=True)
    return _flashrank_ranker if _flashrank_ranker else None


def _parse_repo_topics() -> dict[str, list[str]]:
    """Parse SE_REPO_TOPICS into {repo_name: [keyword, ...]} dict.

    Format: "repo-name:kw1,kw2; repo2:kw3,kw4"
    """
    try:
        from core.config import get_settings

        raw = get_settings().repo_topics.strip()
    except Exception:
        return {}
    result: dict[str, list[str]] = {}
    for segment in raw.split(";"):
        segment = segment.strip()
        if ":" not in segment:
            continue
        name, _, kws = segment.partition(":")
        keywords = [k.strip().lower() for k in kws.split(",") if k.strip()]
        if name.strip() and keywords:
            result[name.strip()] = keywords
    return result


def _score_paper_for_repos(paper: dict, repo_topics: dict[str, list[str]]) -> list[str]:
    """Return a list of repo names whose keywords appear in the paper's text.

    A paper is considered relevant to a repo when at least one of its
    configured keywords matches in the title or summary.
    """
    text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
    matched: list[str] = []
    for repo, keywords in repo_topics.items():
        if any(kw in text for kw in keywords):
            matched.append(repo)
    return matched


def _ranker(state: IntelligenceState) -> dict[str, Any]:
    """Score papers by semantic relevance; return top 8. No LLM call.

    Uses FlashRank cross-encoder when available (recommended). Falls back to
    keyword counting when flashrank is not installed or the query is empty.
    Install: uv add flashrank  (adds ~4 MB ONNX model on first run).

    Also annotates each paper with ``repos`` — a list of local repo names
    that match the paper's content, enabling paper→repo relevance in the brief.
    """
    papers = state["raw_papers"]
    if not papers:
        return {"ranked_papers": [], "repo_relevant_papers": []}

    repo_topics = _parse_repo_topics()

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
        except Exception:  # graceful fallback to keyword scorer
            logger.warning("intel_ranker_flashrank_failed — falling back to keyword", exc_info=True)
            ranked = _keyword_rank(papers)
    else:
        ranked = _keyword_rank(papers)

    # Annotate each ranked paper with matching repo names (non-destructive copy)
    repo_relevant: list[dict] = []
    annotated: list[dict] = []
    for paper in ranked:
        matched_repos = _score_paper_for_repos(paper, repo_topics) if repo_topics else []
        annotated_paper = {**paper, "repos": matched_repos}
        annotated.append(annotated_paper)
        if matched_repos:
            repo_relevant.append(annotated_paper)

    logger.info(
        "intel_ranker_repo_hits papers=%d repo_relevant=%d",
        len(annotated),
        len(repo_relevant),
    )
    return {"ranked_papers": annotated, "repo_relevant_papers": repo_relevant}


def _keyword_rank(papers: list[dict]) -> list[dict]:
    """Rank papers by keyword hit count. Returns top 8."""

    def _score(paper: dict) -> int:
        text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
        return sum(1 for kw in _RELEVANCE_KEYWORDS if kw in text)

    ranked = sorted(papers, key=_score, reverse=True)
    top = ranked[:8]
    logger.info("intel_ranker_keyword input=%d top=%d", len(papers), len(top))
    return top


async def _synthesizer(state: IntelligenceState) -> dict[str, Any]:
    """Generate the intelligence brief from ranked papers via LLM."""
    from llm.gateway import get_gateway

    gateway = get_gateway()

    # Build paper context string
    research_context = ""
    if state["ranked_papers"]:
        try:
            from search.arxiv import format_papers

            research_context = "Recent AI/ML papers:\n" + format_papers(state["ranked_papers"])
        except Exception:
            logger.warning("intel_format_papers_failed", exc_info=True)

    # Build repo-relevance annotation block
    repo_context = ""
    if state.get("repo_relevant_papers"):
        lines: list[str] = ["Papers that match your local repos:"]
        for p in state["repo_relevant_papers"]:
            repos = ", ".join(p.get("repos", []))
            lines.append(f"  • {p.get('title', '')} → [{repos}]")
        repo_context = "\n".join(lines)

    if state["is_morning_brief"]:
        parts = ["Live research data:\n" + research_context] if research_context else []
        if repo_context:
            parts.append(repo_context)
        parts.append(MORNING_PROMPT)
        user_content = "\n\n---\n".join(parts)
        max_tokens = 400  # bumped from 300 to fit repo-relevance section
    else:
        user_input = f"<user_request>\n{state['query']}\n</user_request>"
        parts = ["Live research data:\n" + research_context] if research_context else []
        if repo_context:
            parts.append(repo_context)
        parts.append(user_input)
        user_content = "\n\n---\n".join(parts)
        max_tokens = 2048

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state["history"],
        {"role": "user", "content": user_content},
    ]

    # ── Structured path (instructor + Pydantic) ────────────────────────────
    structured = await gateway.complete_structured(
        messages=messages,
        response_model=IntelBriefResponse,
        routing=state["routing"],
        max_tokens=max_tokens,
        temperature=0.3,
        expert="intelligence",
    )
    if structured is not None:
        logger.info("intel_synthesizer_structured papers=%d", len(structured.papers))
        return {
            "response": format_intel_brief(structured, state.get("repo_relevant_papers", [])),
            "model_used": "structured",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }

    # ── Unstructured fallback ──────────────────────────────────────────────
    logger.warning("intel_synthesizer_structured_failed — falling back to unstructured")
    result = await gateway.complete(
        messages=messages,
        max_tokens=max_tokens,
        routing=state["routing"],
        expert="intelligence",
    )

    brief = BriefOutput(content=result)
    if not brief.is_valid:
        logger.warning(
            "intel_brief_quality_low links=%d words=%d",
            brief.link_count,
            brief.word_count,
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
