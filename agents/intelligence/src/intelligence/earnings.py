"""Earnings context enrichment for Sovereign Edge Intelligence Squad.

Two-layer architecture:
  Layer 1 (free, no key): yFinance earnings_dates + news for EPS data and headlines.
  Layer 2 (optional): FMP free tier (250 req/day) — full transcript → LLM 3-bullet summary.

Set FMP_API_KEY in .env to enable transcript summaries.  Without a key the module
still delivers EPS beat/miss context and the latest news headlines for each symbol.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

import structlog
from core.config import Settings
from llm.gateway import LLMGateway, Message

from intelligence.market import Quote

logger = structlog.get_logger(__name__)

_FMP_BASE = "https://financialmodelingprep.com/api/v3/earning_call_transcript"


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class EarningsContext:
    symbol: str
    eps_actual: float | None = None
    eps_estimate: float | None = None
    eps_surprise_pct: float | None = None  # positive = beat, negative = miss
    last_earnings_date: str = ""
    next_earnings_date: str = ""
    recent_headlines: list[str] = field(default_factory=list)
    transcript_summary: str = ""  # populated by Layer 2 (FMP + LLM)


# ── Layer 1: yFinance EPS + headlines ─────────────────────────────────────────


def _parse_earnings_sync(symbol: str) -> EarningsContext:
    """Fetch EPS history and recent news via yFinance (synchronous).

    Call via asyncio.to_thread — yFinance is blocking.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.warning("intelligence.earnings.yfinance_not_installed")
        return EarningsContext(symbol=symbol)

    ctx = EarningsContext(symbol=symbol)
    t = yf.Ticker(symbol)

    # EPS data — earnings_dates DataFrame indexed by timezone-aware datetime
    try:
        import pandas as pd  # yfinance transitive dependency — always present

        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            now = pd.Timestamp.now(tz="UTC")
            past = ed[ed.index < now]
            if not past.empty:
                row = past.iloc[0]
                try:
                    val = float(row["Reported EPS"])
                    ctx.eps_actual = None if math.isnan(val) else val
                except (KeyError, TypeError, ValueError):
                    pass
                try:
                    val = float(row["EPS Estimate"])
                    ctx.eps_estimate = None if math.isnan(val) else val
                except (KeyError, TypeError, ValueError):
                    pass
                try:
                    val = float(row["Surprise(%)"])
                    ctx.eps_surprise_pct = None if math.isnan(val) else round(val, 1)
                except (KeyError, TypeError, ValueError):
                    pass
                ctx.last_earnings_date = str(past.index[0].date())
            future = ed[ed.index >= now]
            if not future.empty:
                ctx.next_earnings_date = str(future.index[-1].date())
    except Exception:
        logger.warning("intelligence.earnings.eps_parse_failed", symbol=symbol, exc_info=True)

    # Latest news headlines (top 3)
    try:
        news = t.news or []
        ctx.recent_headlines = [item.get("title", "") for item in news[:5] if item.get("title")][:3]
    except Exception:
        logger.warning("intelligence.earnings.news_failed", symbol=symbol, exc_info=True)

    logger.info(
        "intelligence.earnings.parsed",
        symbol=symbol,
        eps_surprise_pct=ctx.eps_surprise_pct,
        next_earnings=ctx.next_earnings_date,
    )
    return ctx


async def get_earnings_context(symbol: str) -> EarningsContext:
    """Async wrapper — runs yFinance in a thread pool."""
    return await asyncio.to_thread(_parse_earnings_sync, symbol)


# ── Layer 2: FMP transcript → LLM summary ────────────────────────────────────


def _fetch_transcript_sync(symbol: str, fmp_api_key: str) -> str:
    """Fetch the most recent earnings call transcript from FMP.

    Tries the current quarter and walks back up to 4 quarters.
    Returns empty string when no transcript is found or on error.
    FMP free tier: 250 requests/day.
    """
    import datetime

    try:
        import httpx  # type: ignore
    except ImportError:
        logger.warning("intelligence.earnings.httpx_not_installed")
        return ""

    now = datetime.date.today()
    year = now.year
    quarter = (now.month - 1) // 3 + 1

    for _ in range(4):
        url = f"{_FMP_BASE}/{symbol}?quarter={quarter}&year={year}&apikey={fmp_api_key}"
        try:
            resp = httpx.get(url, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list) and data[0].get("content"):
                logger.info(
                    "intelligence.earnings.transcript_fetched",
                    symbol=symbol,
                    quarter=quarter,
                    year=year,
                )
                return str(data[0]["content"])
        except Exception:
            logger.warning(
                "intelligence.earnings.transcript_fetch_failed",
                symbol=symbol,
                quarter=quarter,
                year=year,
                exc_info=True,
            )
        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1

    return ""


async def get_transcript_summary(
    symbol: str,
    *,
    fmp_api_key: str,
    llm: LLMGateway,
) -> str:
    """Fetch transcript from FMP and return an LLM 3-bullet summary.

    Returns empty string when fmp_api_key is absent or no transcript is found.
    """
    if not fmp_api_key:
        return ""

    transcript = await asyncio.to_thread(_fetch_transcript_sync, symbol, fmp_api_key)
    if not transcript:
        return ""

    trimmed = transcript[:4000]  # ~1k tokens — enough context, avoids overflow
    system = (
        "Summarize this earnings call in exactly 3 concise bullet points. "
        "Cover: (1) revenue/EPS vs expectations, "
        "(2) key growth drivers or risks, "
        "(3) management guidance or outlook. Each bullet ≤ 20 words."
    )
    return await llm.complete(
        [Message.user(f"Earnings call transcript for {symbol}:\n\n{trimmed}")],
        system=system,
        max_tokens=200,
    )


# ── Formatting ────────────────────────────────────────────────────────────────


def format_earnings_one_liner(ctx: EarningsContext) -> str:
    """Compact single-line earnings summary for Telegram output.

    Example: "EPS +8.3% beat | Next: 2025-07-28 | '• Revenue beat expectations'"
    """
    parts: list[str] = []

    if ctx.eps_surprise_pct is not None:
        direction = "beat" if ctx.eps_surprise_pct >= 0 else "miss"
        parts.append(f"EPS {ctx.eps_surprise_pct:+.1f}% {direction}")

    if ctx.next_earnings_date:
        parts.append(f"Next: {ctx.next_earnings_date}")

    if ctx.transcript_summary:
        first_bullet = ctx.transcript_summary.strip().split("\n")[0][:80]
        parts.append(first_bullet)
    elif ctx.recent_headlines:
        parts.append(f"'{ctx.recent_headlines[0][:60]}'")

    return " | ".join(parts)


# ── Enrichment ────────────────────────────────────────────────────────────────


async def enrich_quotes_with_earnings(
    quotes: list[Quote],
    *,
    settings: Settings,
    llm: LLMGateway | None = None,
) -> list[Quote]:
    """Populate Quote.earnings_context for each quote in-place.

    Layer 1 (always free): yFinance EPS beat/miss + news headlines.
    Layer 2 (optional): FMP transcript → LLM 3-bullet summary when fmp_api_key is set.
    """
    fmp_key = settings.fmp_api_key
    for q in quotes:
        try:
            ctx = await get_earnings_context(q.symbol)
            if fmp_key and llm:
                ctx.transcript_summary = await get_transcript_summary(
                    q.symbol, fmp_api_key=fmp_key, llm=llm
                )
            q.earnings_context = format_earnings_one_liner(ctx)
        except Exception:
            logger.error("intelligence.earnings.enrich_failed", symbol=q.symbol, exc_info=True)
    return quotes
