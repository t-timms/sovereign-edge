"""
Multi-source job fetcher — The Muse (free) + Remotive (free) + Adzuna (free 50 RPD).

The Muse and Remotive require no authentication.
Adzuna requires a free API key (developer.adzuna.com — 2-minute signup).
All sources are fetched in parallel via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_MUSE_BASE = "https://www.themuse.com/api/public/jobs"
_REMOTIVE_BASE = "https://remotive.com/api/remote-jobs"
_ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs/us/search/1"

_ML_TITLE_KEYWORDS: frozenset[str] = frozenset(
    {
        "machine learning",
        "ml engineer",
        "ai engineer",
        "llm engineer",
        "data scientist",
        "deep learning",
        "nlp engineer",
        "mlops",
        "ml ops",
        "ai/ml",
        "artificial intelligence",
        "research scientist",
        "applied scientist",
        "computer vision",
        "generative ai",
        "large language",
        "foundation model",
        "inference engineer",
        "platform engineer",
    }
)


@dataclass
class JobRawListing:
    """Normalized job listing from any source — before LLM structuring."""

    company: str
    title: str
    location: str
    apply_url: str
    source: str
    salary: str = ""
    description_snippet: str = ""


def _is_ml_relevant(title: str) -> bool:
    """True if the job title contains any ML/AI keyword."""
    t = title.lower()
    return any(kw in t for kw in _ML_TITLE_KEYWORDS)


# ── The Muse ──────────────────────────────────────────────────────────────────


async def _fetch_muse(client: httpx.AsyncClient) -> list[JobRawListing]:
    """Fetch ML-relevant jobs from The Muse (no auth required)."""
    results: list[JobRawListing] = []
    query_params = [
        {"category": "Data Science", "location": "Dallas, TX", "page": 1, "descending": "true"},
        {"category": "Data Science", "location": "Remote", "page": 1, "descending": "true"},
        {"category": "Engineering", "location": "Dallas, TX", "page": 1, "descending": "true"},
        {"category": "Engineering", "location": "Remote", "page": 1, "descending": "true"},
        {"category": "Data & Analytics", "location": "Dallas, TX", "page": 1, "descending": "true"},
    ]
    for params in query_params:
        try:
            resp = await client.get(_MUSE_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
            for job in data.get("results", []):
                title: str = job.get("name", "")
                if not _is_ml_relevant(title):
                    continue
                company: str = job.get("company", {}).get("name", "")
                url: str = job.get("refs", {}).get("landing_page", "")
                if not title or not company or not url:
                    continue
                locs = [loc.get("name", "") for loc in job.get("locations", [])]
                results.append(
                    JobRawListing(
                        company=company,
                        title=title,
                        location=locs[0] if locs else str(params.get("location", "")),
                        apply_url=url,
                        source="the_muse",
                    )
                )
        except (httpx.HTTPError, KeyError, ValueError):
            logger.warning("muse_fetch_failed params=%s", params, exc_info=True)
    logger.info("jobs_muse_fetched count=%d", len(results))
    return results


# ── Remotive ──────────────────────────────────────────────────────────────────


async def _fetch_remotive(client: httpx.AsyncClient, search: str) -> list[JobRawListing]:
    """Fetch remote ML/AI jobs from Remotive (no auth required)."""
    results: list[JobRawListing] = []
    try:
        resp = await client.get(
            _REMOTIVE_BASE,
            params={"category": "software-dev", "search": search, "limit": 25},
        )
        resp.raise_for_status()
        data = resp.json()
        for job in data.get("jobs", []):
            title: str = job.get("title", "")
            if not _is_ml_relevant(title):
                continue
            results.append(
                JobRawListing(
                    company=job.get("company_name", ""),
                    title=title,
                    location=job.get("candidate_required_location", "Worldwide / Remote"),
                    apply_url=job.get("url", ""),
                    source="remotive",
                    salary=job.get("salary", ""),
                    description_snippet=(job.get("description", "") or "")[:400],
                )
            )
    except (httpx.HTTPError, KeyError, ValueError):
        logger.warning("remotive_fetch_failed search=%r", search, exc_info=True)
    logger.info("jobs_remotive_fetched search=%r count=%d", search, len(results))
    return results


# ── Adzuna ────────────────────────────────────────────────────────────────────


async def _fetch_adzuna(
    client: httpx.AsyncClient,
    app_id: str,
    app_key: str,
    query: str = "machine learning engineer",
    where: str = "dallas",
) -> list[JobRawListing]:
    """Fetch from Adzuna (free tier: 50 req/day — developer.adzuna.com)."""
    if not app_id or not app_key:
        return []
    results: list[JobRawListing] = []
    try:
        resp = await client.get(
            _ADZUNA_BASE,
            params={
                "app_id": app_id,
                "app_key": app_key,
                "what": query,
                "where": where,
                "distance": 50,
                "results_per_page": 20,
                "sort_by": "date",
                "max_days_old": 7,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        for job in data.get("results", []):
            salary_min = job.get("salary_min")
            salary_max = job.get("salary_max")
            salary = f"${salary_min:,.0f}-${salary_max:,.0f}" if salary_min and salary_max else ""
            results.append(
                JobRawListing(
                    company=job.get("company", {}).get("display_name", ""),
                    title=job.get("title", ""),
                    location=job.get("location", {}).get("display_name", where.title()),
                    apply_url=job.get("redirect_url", ""),
                    source="adzuna",
                    salary=salary,
                    description_snippet=(job.get("description", "") or "")[:400],
                )
            )
    except (httpx.HTTPError, KeyError, ValueError):
        logger.warning("adzuna_fetch_failed query=%r where=%r", query, where, exc_info=True)
    logger.info("jobs_adzuna_fetched count=%d", len(results))
    return results


# ── Public API ────────────────────────────────────────────────────────────────


async def fetch_all_sources(
    adzuna_app_id: str = "",
    adzuna_app_key: str = "",
) -> list[JobRawListing]:
    """Fetch from all free job sources in parallel.

    Returns a combined, unfiltered list.
    Deduplication is handled downstream by job_store.JobStore.filter_new().
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        tasks: list = [
            _fetch_muse(client),
            _fetch_remotive(client, "machine learning engineer"),
            _fetch_remotive(client, "LLM AI engineer"),
        ]
        if adzuna_app_id and adzuna_app_key:
            tasks.extend(
                [
                    _fetch_adzuna(
                        client, adzuna_app_id, adzuna_app_key, "machine learning engineer"
                    ),
                    _fetch_adzuna(
                        client, adzuna_app_id, adzuna_app_key, "AI engineer LLM", "plano"
                    ),
                ]
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

    combined: list[JobRawListing] = []
    for r in results:
        if isinstance(r, list):
            combined.extend(r)
        elif isinstance(r, Exception):
            logger.warning("jobs_source_error error=%s", r)

    logger.info("jobs_all_sources_total count=%d", len(combined))
    return combined


def format_raw_listings(listings: list[JobRawListing]) -> str:
    """Format a list of JobRawListings as LLM-readable context."""
    if not listings:
        return ""
    lines: list[str] = [f"LIVE JOB LISTINGS — {len(listings)} new positions not seen in 7 days:"]
    for i, job in enumerate(listings, 1):
        salary_part = f" | {job.salary}" if job.salary else ""
        lines.append(f"{i}. {job.company} — {job.title} | {job.location}{salary_part}")
        lines.append(f"   Apply: {job.apply_url}")
        if job.description_snippet:
            lines.append(f"   Snippet: {job.description_snippet[:200]}")
    return "\n".join(lines)
