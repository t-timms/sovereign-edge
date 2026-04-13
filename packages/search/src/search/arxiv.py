"""
arXiv paper fetching — 100% free, no API key required.

Uses the arXiv Atom API: https://export.arxiv.org/api/query
Rate limit: 3 requests/second per IP (we're well within that).

Production upgrades:
  - Module-level persistent AsyncClient (connection pooling)
  - 2-retry exponential backoff for transient failures
  - Daily novelty filter — tracks seen paper IDs so the 04:30 prefetch
    and the 05:30 morning brief never surface duplicate papers
  - aclose() for clean shutdown
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from urllib.parse import urlencode

import defusedxml.ElementTree as ET  # type: ignore[import-untyped]
import httpx

logger = logging.getLogger(__name__)

_BASE = "https://export.arxiv.org/api/query"
_TIMEOUT = 20.0
_NS = {"atom": "http://www.w3.org/2005/Atom"}
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 1.0

# Default query targets AI/ML/NLP/CV — the most relevant categories for this system
_DEFAULT_QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"


# ── Persistent client ──────────────────────────────────────────────────────────
_client: httpx.AsyncClient | None = None
_client_lock: asyncio.Lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _client
    async with _client_lock:
        if _client is None:
            _client = httpx.AsyncClient(
                timeout=_TIMEOUT,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
    return _client


# ── Daily novelty filter ───────────────────────────────────────────────────────
# Prevents the same paper appearing in both the 04:30 prefetch and 05:30 brief.
# Resets automatically at midnight.
_seen_ids: set[str] = set()
_seen_date: date = date.today()


def _get_seen_ids() -> set[str]:
    """Return today's seen-IDs set, resetting at midnight."""
    global _seen_ids, _seen_date
    today = date.today()
    if _seen_date != today:
        _seen_ids = set()
        _seen_date = today
        logger.debug("arxiv_novelty_filter_reset")
    return _seen_ids


# ── Public API ─────────────────────────────────────────────────────────────────


async def fetch_recent(
    query: str = _DEFAULT_QUERY,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Return the most recently submitted papers matching *query*.

    Papers already surfaced today (seen by the novelty filter) are excluded
    from results and the new ones are registered as seen.
    """
    params = urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results * 2,  # fetch extra to account for novelty filter
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    client = await _get_client()
    resp_text: str | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get(f"{_BASE}?{params}")
            resp.raise_for_status()
            resp_text = resp.text
            break
        except httpx.HTTPError as exc:
            logger.warning("arxiv_fetch_failed attempt=%d error=%s", attempt + 1, exc)
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))

    if resp_text is None:
        return []

    try:
        root = ET.fromstring(resp_text)
    except ET.ParseError as exc:
        logger.warning("arxiv_xml_parse_error error=%s", exc)
        return []

    seen = _get_seen_ids()
    papers: list[dict[str, str]] = []

    for entry in root.findall("atom:entry", _NS):
        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)
        id_el = entry.find("atom:id", _NS)
        authors = [
            a.findtext("atom:name", default="", namespaces=_NS)
            for a in entry.findall("atom:author", _NS)
        ]
        title = (title_el.text or "").strip().replace("\n", " ")
        summary = (summary_el.text or "").strip().replace("\n", " ")[:400]
        url = (id_el.text or "").strip().replace("http://arxiv.org", "https://arxiv.org")
        # Extract short arXiv ID for the novelty key (e.g. "2503.12345")
        arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else url

        if not title or arxiv_id in seen:
            continue

        seen.add(arxiv_id)
        papers.append(
            {
                "title": title,
                "authors": ", ".join(a for a in authors[:3] if a),
                "summary": summary,
                "url": url,
            }
        )

        if len(papers) >= max_results:
            break

    logger.debug("arxiv_fetch_ok count=%d", len(papers))
    return papers


def format_papers(papers: list[dict[str, str]]) -> str:
    """Render paper list as concise markdown suitable for LLM context.

    Handles both arXiv papers (with 'authors') and HuggingFace papers
    (with 'upvotes') so the intelligence ranker's merged list always formats
    cleanly regardless of source.
    """
    if not papers:
        return ""
    lines: list[str] = []
    for p in papers:
        lines.append(f"**{p['title']}**")
        if authors := p.get("authors", ""):
            lines.append(f"*{authors}*")
        elif upvotes := p.get("upvotes", ""):
            lines.append(f"*HF Community Paper — ↑{upvotes} votes*")
        lines.append(p.get("summary", ""))
        lines.append(f"→ {p.get('url', '')}")
        lines.append("")
    return "\n".join(lines).strip()


async def aclose() -> None:
    """Close the persistent HTTP client. Call during application shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
