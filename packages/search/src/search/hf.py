"""
Hugging Face Daily Papers API — free, no auth required.

Returns papers curated and upvoted by the HF community each day.
Endpoint: https://huggingface.co/api/daily_papers

Production upgrades:
  - Module-level persistent AsyncClient (connection pooling)
  - 2-retry exponential backoff for transient failures
  - aclose() for clean shutdown
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://huggingface.co/api/daily_papers"
_TIMEOUT = 15.0
_MAX_PAPERS = 5
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 1.0


# ── Persistent client ──────────────────────────────────────────────────────────
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=3, max_connections=5),
        )
    return _client


# ── Public API ─────────────────────────────────────────────────────────────────


async def fetch_daily_papers(for_date: date | None = None) -> list[dict[str, str]]:
    """Fetch today's (or a specific day's) featured HF papers."""
    params: dict[str, str] = {}
    if for_date:
        params["date"] = for_date.isoformat()

    client = _get_client()

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get(_BASE, params=params)
            resp.raise_for_status()
            items: list[dict] = resp.json()
            break
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("hf_daily_papers_failed attempt=%d error=%s", attempt + 1, exc)
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))
    else:
        return []

    papers: list[dict[str, str]] = []
    for item in items[:_MAX_PAPERS]:
        paper = item.get("paper", {})
        title = (paper.get("title") or "").strip()
        summary = (paper.get("summary") or "").strip()[:350]
        paper_id = paper.get("id", "")
        upvotes = str(paper.get("upvotes") or 0)  # 0 when field absent; never use numComments
        if title and paper_id:
            papers.append(
                {
                    "title": title,
                    "summary": summary,
                    "url": f"https://huggingface.co/papers/{paper_id}",
                    "upvotes": upvotes,
                }
            )

    logger.debug("hf_daily_papers_ok count=%d", len(papers))
    return papers


def format_hf_papers(papers: list[dict[str, str]]) -> str:
    """Format HF paper list as concise markdown for LLM context."""
    if not papers:
        return ""
    lines: list[str] = []
    for p in papers:
        lines.append(f"**{p['title']}** (↑{p['upvotes']})")
        if p["summary"]:
            lines.append(p["summary"])
        lines.append(f"→ {p['url']}")
        lines.append("")
    return "\n".join(lines).strip()


async def aclose() -> None:
    """Close the persistent HTTP client. Call during application shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
