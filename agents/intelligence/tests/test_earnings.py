"""Tests for earnings.py — EPS context, transcript fetch, formatting, enrichment."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from intelligence.earnings import (
    EarningsContext,
    _fetch_transcript_sync,
    enrich_quotes_with_earnings,
    format_earnings_one_liner,
    get_earnings_context,
    get_transcript_summary,
)
from intelligence.market import Quote

# ── EarningsContext defaults ───────────────────────────────────────────────────


def test_earnings_context_defaults() -> None:
    ctx = EarningsContext(symbol="AAPL")
    assert ctx.symbol == "AAPL"
    assert ctx.eps_actual is None
    assert ctx.eps_estimate is None
    assert ctx.eps_surprise_pct is None
    assert ctx.last_earnings_date == ""
    assert ctx.next_earnings_date == ""
    assert ctx.recent_headlines == []
    assert ctx.transcript_summary == ""


# ── format_earnings_one_liner ─────────────────────────────────────────────────


def test_format_earnings_one_liner_beat() -> None:
    ctx = EarningsContext(symbol="NVDA", eps_surprise_pct=8.3, next_earnings_date="2025-07-28")
    result = format_earnings_one_liner(ctx)
    assert "EPS +8.3% beat" in result
    assert "Next: 2025-07-28" in result


def test_format_earnings_one_liner_miss() -> None:
    ctx = EarningsContext(symbol="GOOGL", eps_surprise_pct=-3.1)
    result = format_earnings_one_liner(ctx)
    assert "EPS -3.1% miss" in result


def test_format_earnings_one_liner_no_eps() -> None:
    ctx = EarningsContext(symbol="MSFT", next_earnings_date="2025-10-22")
    result = format_earnings_one_liner(ctx)
    assert "EPS" not in result
    assert "Next: 2025-10-22" in result


def test_format_earnings_one_liner_empty() -> None:
    ctx = EarningsContext(symbol="SPY")
    assert format_earnings_one_liner(ctx) == ""


def test_format_earnings_headline_fallback() -> None:
    ctx = EarningsContext(symbol="META", recent_headlines=["AI ad revenue surges 40%"])
    result = format_earnings_one_liner(ctx)
    assert "AI ad revenue surges 40%" in result


def test_format_earnings_transcript_overrides_headline() -> None:
    ctx = EarningsContext(
        symbol="META",
        recent_headlines=["Old headline"],
        transcript_summary=(
            "• Revenue beat by 12%\n• AI investments accelerating\n• Raised guidance"
        ),
    )
    result = format_earnings_one_liner(ctx)
    assert "Revenue beat by 12%" in result
    assert "Old headline" not in result


def test_format_earnings_headline_truncated_at_60() -> None:
    long_headline = "X" * 80
    ctx = EarningsContext(symbol="T", recent_headlines=[long_headline])
    result = format_earnings_one_liner(ctx)
    # Headline is clipped to 60 chars inside quotes
    assert len(result) < 75  # 1 quote + 60 chars + 1 quote + pipe separators


# ── get_earnings_context ──────────────────────────────────────────────────────


async def test_get_earnings_context_delegates_to_sync() -> None:
    expected = EarningsContext(symbol="AAPL", eps_surprise_pct=5.2, next_earnings_date="2025-10-30")
    with patch("intelligence.earnings._parse_earnings_sync", return_value=expected) as mock_sync:
        ctx = await get_earnings_context("AAPL")
    assert ctx.eps_surprise_pct == 5.2
    assert ctx.next_earnings_date == "2025-10-30"
    mock_sync.assert_called_once_with("AAPL")


# ── _fetch_transcript_sync ────────────────────────────────────────────────────


def test_fetch_transcript_sync_no_httpx() -> None:
    with patch.dict("sys.modules", {"httpx": None}):  # type: ignore[arg-type]
        result = _fetch_transcript_sync("AAPL", "fake_key")
    assert result == ""


def test_fetch_transcript_sync_success() -> None:
    mock_httpx = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"content": "Good morning, this is the Q4 call..."}]
    mock_httpx.get.return_value = mock_resp

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        result = _fetch_transcript_sync("NVDA", "real_key")

    assert result == "Good morning, this is the Q4 call..."
    mock_httpx.get.assert_called_once()


def test_fetch_transcript_sync_empty_response_walks_back() -> None:
    mock_httpx = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = []  # No transcript for any quarter
    mock_httpx.get.return_value = mock_resp

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        result = _fetch_transcript_sync("AAPL", "real_key")

    assert result == ""
    assert mock_httpx.get.call_count == 4  # tried all 4 quarters


# ── get_transcript_summary ────────────────────────────────────────────────────


async def test_get_transcript_summary_no_key() -> None:
    llm = MagicMock()
    result = await get_transcript_summary("AAPL", fmp_api_key="", llm=llm)
    assert result == ""
    llm.complete.assert_not_called()


async def test_get_transcript_summary_no_transcript() -> None:
    llm = AsyncMock()
    with patch("intelligence.earnings._fetch_transcript_sync", return_value=""):
        result = await get_transcript_summary("AAPL", fmp_api_key="key123", llm=llm)
    assert result == ""
    llm.complete.assert_not_called()


async def test_get_transcript_summary_with_transcript() -> None:
    llm = AsyncMock()
    llm.complete.return_value = "• Revenue beat\n• AI growth strong\n• Raised guidance"
    with patch("intelligence.earnings._fetch_transcript_sync", return_value="Q4 transcript text"):
        result = await get_transcript_summary("NVDA", fmp_api_key="key123", llm=llm)
    assert "Revenue beat" in result
    llm.complete.assert_called_once()


# ── enrich_quotes_with_earnings ───────────────────────────────────────────────


async def test_enrich_quotes_no_fmp_key() -> None:
    quote = Quote(symbol="AAPL", price=150.0, change_pct=1.0, volume=1_000_000, timestamp="")
    ctx = EarningsContext(symbol="AAPL", eps_surprise_pct=5.2, next_earnings_date="2025-07-28")

    settings = MagicMock()
    settings.fmp_api_key = ""

    with patch("intelligence.earnings._parse_earnings_sync", return_value=ctx):
        quotes = await enrich_quotes_with_earnings([quote], settings=settings)

    assert "EPS +5.2% beat" in quotes[0].earnings_context
    assert "Next: 2025-07-28" in quotes[0].earnings_context


async def test_enrich_quotes_with_fmp_key_calls_llm() -> None:
    quote = Quote(symbol="NVDA", price=900.0, change_pct=2.5, volume=5_000_000, timestamp="")
    ctx = EarningsContext(symbol="NVDA", eps_surprise_pct=12.0)
    llm = AsyncMock()
    llm.complete.return_value = "• Beat by 12%\n• Data center boom\n• Raised FY guidance"

    settings = MagicMock()
    settings.fmp_api_key = "secret"

    with (
        patch("intelligence.earnings._parse_earnings_sync", return_value=ctx),
        patch("intelligence.earnings._fetch_transcript_sync", return_value="transcript text"),
    ):
        quotes = await enrich_quotes_with_earnings([quote], settings=settings, llm=llm)

    assert "Beat by 12%" in quotes[0].earnings_context
    llm.complete.assert_called_once()


async def test_enrich_quotes_error_resilience() -> None:
    """A failing symbol does not prevent enrichment of subsequent symbols."""
    q1 = Quote(symbol="AAPL", price=150.0, change_pct=1.0, volume=1_000_000, timestamp="")
    q2 = Quote(symbol="BAD", price=0.0, change_pct=0.0, volume=0, timestamp="")

    def _side_effect(symbol: str) -> EarningsContext:
        if symbol == "BAD":
            raise RuntimeError("yFinance exploded")
        return EarningsContext(symbol=symbol, eps_surprise_pct=2.0)

    settings = MagicMock()
    settings.fmp_api_key = ""

    with patch("intelligence.earnings._parse_earnings_sync", side_effect=_side_effect):
        quotes = await enrich_quotes_with_earnings([q1, q2], settings=settings)

    assert "EPS +2.0% beat" in quotes[0].earnings_context
    assert quotes[1].earnings_context == ""  # Error caught; default preserved


async def test_enrich_quotes_empty_list() -> None:
    settings = MagicMock()
    settings.fmp_api_key = ""
    result = await enrich_quotes_with_earnings([], settings=settings)
    assert result == []
