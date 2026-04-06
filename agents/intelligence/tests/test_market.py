"""Tests for market.py — indicators, signals, quotes, alerts, formatting."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from intelligence.market import (
    Quote,
    _compute_atr_pct,
    _compute_bb_position,
    _compute_rsi,
    _compute_signal,
    _compute_volume_ratio,
    _detect_regime,
    _fetch_quotes_sync,
    format_market_summary,
    get_quotes,
    get_watchlist_alerts,
)

# ── Quote dataclass ───────────────────────────────────────────────────────────


def test_quote_fields_positional() -> None:
    q = Quote(symbol="NVDA", price=900.0, change_pct=2.5, volume=5_000_000, timestamp="")
    assert q.symbol == "NVDA"
    assert q.price == 900.0
    assert q.change_pct == 2.5


def test_quote_defaults() -> None:
    q = Quote(symbol="X", price=1.0, change_pct=0.0, volume=0, timestamp="")
    assert q.rsi == 50.0
    assert q.bb_position == 0.0
    assert q.signal == "NEUTRAL"
    assert q.signal_confidence == 0.0
    assert q.regime == "UNKNOWN"


# ── _compute_rsi ──────────────────────────────────────────────────────────────


def test_rsi_insufficient_data_returns_50() -> None:
    assert _compute_rsi([100.0, 101.0]) == 50.0


def test_rsi_all_gains_returns_100() -> None:
    # All positive moves → avg_loss = 0 → RSI = 100
    prices = [float(i) for i in range(1, 17)]  # 16 prices, 15 changes all positive
    assert _compute_rsi(prices) == 100.0


def test_rsi_oversold_range() -> None:
    # Falling prices should produce RSI < 30
    prices = [100.0 - i * 2 for i in range(20)]  # monotonically falling
    rsi = _compute_rsi(prices)
    assert rsi < 30.0


def test_rsi_overbought_range() -> None:
    # Rising prices should produce RSI > 70
    prices = [50.0 + i * 2 for i in range(20)]  # monotonically rising
    rsi = _compute_rsi(prices)
    assert rsi > 70.0


def test_rsi_neutral_range() -> None:
    # Alternating up/down → near 50
    prices = [100.0 + (1 if i % 2 == 0 else -1) for i in range(20)]
    rsi = _compute_rsi(prices)
    assert 40.0 < rsi < 60.0


# ── _compute_bb_position ──────────────────────────────────────────────────────


def test_bb_position_insufficient_data() -> None:
    assert _compute_bb_position([100.0, 101.0]) == 0.0


def test_bb_position_at_mean_is_zero() -> None:
    prices = [100.0] * 20
    assert _compute_bb_position(prices) == 0.0  # std dev = 0


def test_bb_position_above_mean_is_positive() -> None:
    prices = [100.0] * 19 + [110.0]  # last price is well above mean
    pos = _compute_bb_position(prices)
    assert pos > 0.0


def test_bb_position_below_mean_is_negative() -> None:
    prices = [100.0] * 19 + [85.0]  # last price is well below mean
    pos = _compute_bb_position(prices)
    assert pos < 0.0


def test_bb_position_clamped_to_minus_one() -> None:
    # Extreme move far below mean → clamped at -1
    prices = [100.0] * 19 + [1.0]
    assert _compute_bb_position(prices) == -1.0


def test_bb_position_clamped_to_plus_one() -> None:
    # Extreme move far above mean → clamped at +1
    prices = [100.0] * 19 + [999.0]
    assert _compute_bb_position(prices) == 1.0


# ── _compute_atr_pct ─────────────────────────────────────────────────────────


def test_atr_pct_insufficient_data() -> None:
    assert _compute_atr_pct([110.0], [90.0], [100.0]) == 0.0


def test_atr_pct_is_positive() -> None:
    n = 20
    closes = [100.0 + i for i in range(n)]
    highs = [c + 2.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    atr = _compute_atr_pct(highs, lows, closes)
    assert atr > 0.0


def test_atr_pct_flat_market_uses_hl_range() -> None:
    n = 20
    closes = [100.0] * n
    highs = [102.0] * n
    lows = [98.0] * n
    # TR = max(H-L, |H-prev_C|, |L-prev_C|) = max(4, 2, 2) = 4
    # ATR% = 4/100 * 100 = 4.0%
    atr = _compute_atr_pct(highs, lows, closes)
    assert abs(atr - 4.0) < 0.5


# ── _compute_volume_ratio ─────────────────────────────────────────────────────


def test_volume_ratio_single_value_returns_one() -> None:
    assert _compute_volume_ratio([1_000_000]) == 1.0


def test_volume_ratio_spike_above_one() -> None:
    baseline = [1_000_000] * 20
    today = [3_000_000]
    ratio = _compute_volume_ratio(baseline + today)
    assert ratio > 1.5


def test_volume_ratio_quiet_day_below_one() -> None:
    baseline = [1_000_000] * 20
    today = [200_000]
    ratio = _compute_volume_ratio(baseline + today)
    assert ratio < 1.0


# ── _detect_regime ────────────────────────────────────────────────────────────


def test_detect_regime_insufficient_data() -> None:
    assert _detect_regime([100.0], [102.0], [98.0]) == "UNKNOWN"


def test_detect_regime_trending_up() -> None:
    # Rising prices: short MA > long MA by >1%
    closes = [100.0 + i * 0.5 for i in range(25)]  # steady uptrend
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    assert _detect_regime(closes, highs, lows) == "TRENDING_UP"


def test_detect_regime_trending_down() -> None:
    closes = [100.0 - i * 0.5 for i in range(25)]  # steady downtrend
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    assert _detect_regime(closes, highs, lows) == "TRENDING_DOWN"


def test_detect_regime_ranging() -> None:
    # Flat / oscillating prices
    closes = [100.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(25)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    regime = _detect_regime(closes, highs, lows)
    assert regime == "RANGING"


def test_detect_regime_volatile() -> None:
    # High ATR (>5% of price)
    closes = [100.0] * 20
    # H-L range of 12 → ATR ~12 → ATR% ~12% → VOLATILE
    highs = [106.0] * 20
    lows = [94.0] * 20
    assert _detect_regime(closes, highs, lows) == "VOLATILE"


# ── _compute_signal ───────────────────────────────────────────────────────────


def test_signal_mean_reversion_buy() -> None:
    signal, conf = _compute_signal(rsi=28.0, bb_position=-0.85, volume_ratio=1.0, regime="RANGING")
    assert signal == "MEAN_REVERSION_BUY"
    assert conf > 0.6


def test_signal_mean_reversion_sell() -> None:
    signal, conf = _compute_signal(rsi=72.0, bb_position=0.85, volume_ratio=1.0, regime="RANGING")
    assert signal == "MEAN_REVERSION_SELL"
    assert conf > 0.6


def test_signal_momentum_buy() -> None:
    signal, conf = _compute_signal(
        rsi=55.0, bb_position=0.2, volume_ratio=1.6, regime="TRENDING_UP"
    )
    assert signal == "MOMENTUM_BUY"
    assert conf >= 0.55


def test_signal_momentum_sell() -> None:
    signal, conf = _compute_signal(
        rsi=42.0, bb_position=-0.1, volume_ratio=1.6, regime="TRENDING_DOWN"
    )
    assert signal == "MOMENTUM_SELL"
    assert conf >= 0.55


def test_signal_neutral_no_conditions_met() -> None:
    signal, conf = _compute_signal(rsi=52.0, bb_position=0.1, volume_ratio=1.0, regime="RANGING")
    assert signal == "NEUTRAL"
    assert conf == 0.0


def test_signal_mean_reversion_requires_both_conditions() -> None:
    # RSI oversold but BB not extreme → NEUTRAL
    signal, _ = _compute_signal(rsi=28.0, bb_position=-0.5, volume_ratio=1.0, regime="RANGING")
    assert signal == "NEUTRAL"


def test_signal_confidence_capped_at_95_pct() -> None:
    # BB position at maximum (-1) → max confidence, still ≤ 0.95
    signal, conf = _compute_signal(rsi=20.0, bb_position=-1.0, volume_ratio=1.0, regime="RANGING")
    assert signal == "MEAN_REVERSION_BUY"
    assert conf <= 0.95


# ── get_quotes ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_get_quotes_returns_empty_for_empty_symbols() -> None:
    result = await get_quotes([])
    assert result == []


@pytest.mark.asyncio()
async def test_get_quotes_returns_enriched_quote() -> None:
    enriched = [
        Quote(
            symbol="NVDA",
            price=900.0,
            change_pct=2.10,
            volume=1_000_000,
            timestamp="2026-04-03",
            rsi=28.0,
            bb_position=-0.85,
            signal="MEAN_REVERSION_BUY",
            signal_confidence=0.77,
        )
    ]
    with patch("intelligence.market._fetch_quotes_sync", return_value=enriched):
        quotes = await get_quotes(["NVDA"])

    assert len(quotes) == 1
    assert quotes[0].rsi == 28.0
    assert quotes[0].signal == "MEAN_REVERSION_BUY"


@pytest.mark.asyncio()
async def test_get_quotes_returns_empty_when_yfinance_missing() -> None:
    with patch("intelligence.market._fetch_quotes_sync", return_value=[]):
        quotes = await get_quotes(["NVDA"])
    assert quotes == []


@pytest.mark.asyncio()
async def test_get_quotes_continues_on_per_symbol_error() -> None:
    good = Quote(symbol="MSFT", price=400.0, change_pct=0.5, volume=500_000, timestamp="")
    with patch("intelligence.market._fetch_quotes_sync", return_value=[good]):
        quotes = await get_quotes(["NVDA", "MSFT"])

    assert len(quotes) == 1
    assert quotes[0].symbol == "MSFT"


# ── get_watchlist_alerts ──────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_get_watchlist_alerts_filters_by_price_threshold() -> None:
    from core.config import Settings

    big_mover = Quote(symbol="NVDA", price=900.0, change_pct=3.5, volume=1_000_000, timestamp="")
    small_mover = Quote(symbol="MSFT", price=400.0, change_pct=0.5, volume=500_000, timestamp="")
    settings = Settings(market_alert_threshold=0.02, watchlist=["NVDA", "MSFT"])

    with (
        patch("intelligence.market.get_settings", return_value=settings),
        patch(
            "intelligence.market.get_quotes", new=AsyncMock(return_value=[big_mover, small_mover])
        ),
    ):
        alerts = await get_watchlist_alerts()

    assert len(alerts) == 1
    assert alerts[0].symbol == "NVDA"


@pytest.mark.asyncio()
async def test_get_watchlist_alerts_includes_signal_despite_small_move() -> None:
    """A signal overrides the price threshold — setup quality matters more than % move."""
    from core.config import Settings

    signal_quote = Quote(
        symbol="NVDA",
        price=900.0,
        change_pct=0.3,  # tiny move — would normally be ignored
        volume=1_000_000,
        timestamp="",
        rsi=28.0,
        bb_position=-0.85,
        signal="MEAN_REVERSION_BUY",
        signal_confidence=0.77,
    )
    settings = Settings(market_alert_threshold=0.02, watchlist=["NVDA"])

    with (
        patch("intelligence.market.get_settings", return_value=settings),
        patch("intelligence.market.get_quotes", new=AsyncMock(return_value=[signal_quote])),
    ):
        alerts = await get_watchlist_alerts()

    assert len(alerts) == 1
    assert alerts[0].signal == "MEAN_REVERSION_BUY"


@pytest.mark.asyncio()
async def test_get_watchlist_alerts_includes_negative_movers() -> None:
    from core.config import Settings

    dropper = Quote(symbol="META", price=500.0, change_pct=-4.2, volume=2_000_000, timestamp="")
    settings = Settings(market_alert_threshold=0.02, watchlist=["META"])

    with (
        patch("intelligence.market.get_settings", return_value=settings),
        patch("intelligence.market.get_quotes", new=AsyncMock(return_value=[dropper])),
    ):
        alerts = await get_watchlist_alerts()

    assert len(alerts) == 1
    assert alerts[0].symbol == "META"


@pytest.mark.asyncio()
async def test_get_watchlist_alerts_excludes_neutral_small_movers() -> None:
    from core.config import Settings

    quiet = Quote(
        symbol="MSFT",
        price=400.0,
        change_pct=0.5,
        volume=500_000,
        timestamp="",
        signal="NEUTRAL",
    )
    settings = Settings(market_alert_threshold=0.02, watchlist=["MSFT"])

    with (
        patch("intelligence.market.get_settings", return_value=settings),
        patch("intelligence.market.get_quotes", new=AsyncMock(return_value=[quiet])),
    ):
        alerts = await get_watchlist_alerts()

    assert alerts == []


# ── format_market_summary ─────────────────────────────────────────────────────


def test_format_market_summary_empty() -> None:
    result = format_market_summary([])
    assert result == "No market data available."


def test_format_market_summary_includes_symbols() -> None:
    quotes = [
        Quote(symbol="NVDA", price=900.0, change_pct=2.5, volume=1_000_000, timestamp=""),
        Quote(symbol="MSFT", price=400.0, change_pct=-1.2, volume=500_000, timestamp=""),
    ]
    result = format_market_summary(quotes)
    assert "NVDA" in result
    assert "MSFT" in result
    assert "▲" in result
    assert "▼" in result


def test_format_market_summary_shows_rsi() -> None:
    q = Quote(
        symbol="NVDA",
        price=900.0,
        change_pct=1.0,
        volume=1_000_000,
        timestamp="",
        rsi=28.0,
    )
    result = format_market_summary([q])
    assert "RSI" in result
    assert "28" in result


def test_format_market_summary_shows_signal() -> None:
    q = Quote(
        symbol="NVDA",
        price=900.0,
        change_pct=1.0,
        volume=1_000_000,
        timestamp="",
        rsi=28.0,
        bb_position=-0.85,
        signal="MEAN_REVERSION_BUY",
        signal_confidence=0.77,
    )
    result = format_market_summary([q])
    assert "OVERSOLD BOUNCE" in result
    assert "77%" in result


def test_format_market_summary_correct_arrow_for_zero() -> None:
    quotes = [Quote(symbol="FLAT", price=100.0, change_pct=0.0, volume=0, timestamp="")]
    assert "▲" in format_market_summary(quotes)


def test_format_market_summary_shows_volume_spike() -> None:
    q = Quote(
        symbol="META",
        price=500.0,
        change_pct=2.0,
        volume=5_000_000,
        timestamp="",
        volume_ratio=2.3,
    )
    result = format_market_summary([q])
    assert "Vol" in result
    assert "2.3x" in result


def test_format_market_summary_shows_regime() -> None:
    q = Quote(
        symbol="GOOGL",
        price=175.0,
        change_pct=0.5,
        volume=1_000_000,
        timestamp="",
        regime="TRENDING_UP",
    )
    result = format_market_summary([q])
    assert "TREND↑" in result


# ── _fetch_quotes_sync (unit — no network) ────────────────────────────────────


def test_fetch_quotes_sync_returns_empty_when_yfinance_missing() -> None:
    import sys
    from unittest.mock import patch as _patch

    with _patch.dict(sys.modules, {"yfinance": None}):
        result = _fetch_quotes_sync(["NVDA"])
    assert result == []
