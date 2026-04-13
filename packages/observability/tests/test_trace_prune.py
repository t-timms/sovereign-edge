"""Tests for TraceStore.prune() — SQLite storage pruning for SSD longevity.

Covers:
- Old traces are deleted when beyond max_age_days
- Recent traces are preserved
- Return value matches number of deleted rows
- VACUUM is triggered when >1000 rows deleted
- Zero rows deleted when nothing is old
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def trace_store(tmp_path: Path):
    """TraceStore backed by a temp SQLite file."""
    from observability.traces import TraceStore

    mock_settings = MagicMock()
    mock_settings.logs_path = tmp_path

    with patch("observability.traces.get_settings", return_value=mock_settings):
        return TraceStore()


def _insert_trace(store, *, days_ago: int, expert: str = "intelligence") -> None:
    """Insert a trace row with a timestamp *days_ago* days in the past."""
    ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    store.conn.execute(
        """INSERT INTO traces
           (task_id, timestamp, expert, model, tokens_in, tokens_out,
            latency_ms, cost_usd, cached, routing, status, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("task-1", ts, expert, "test-model", 10, 20, 100.0, 0.001, 0, "CLOUD", "success", ""),
    )
    store.conn.commit()


# ── basic pruning ─────────────────────────────────────────────────────────────


def test_prune_deletes_old_traces(trace_store) -> None:
    """Traces older than max_age_days should be deleted."""
    _insert_trace(trace_store, days_ago=100)
    _insert_trace(trace_store, days_ago=50)

    deleted = trace_store.prune(max_age_days=90)

    assert deleted == 1
    remaining = trace_store.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    assert remaining == 1


def test_prune_preserves_recent_traces(trace_store) -> None:
    """Traces newer than max_age_days must not be deleted."""
    _insert_trace(trace_store, days_ago=5)
    _insert_trace(trace_store, days_ago=10)

    deleted = trace_store.prune(max_age_days=30)

    assert deleted == 0
    remaining = trace_store.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    assert remaining == 2


def test_prune_returns_zero_when_empty(trace_store) -> None:
    """Pruning an empty table returns 0."""
    deleted = trace_store.prune(max_age_days=90)
    assert deleted == 0


def test_prune_deletes_all_old_traces(trace_store) -> None:
    """All traces older than cutoff should be removed."""
    for days in (91, 95, 100, 200):
        _insert_trace(trace_store, days_ago=days)
    _insert_trace(trace_store, days_ago=30)

    deleted = trace_store.prune(max_age_days=90)

    assert deleted == 4
    remaining = trace_store.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    assert remaining == 1


# ── VACUUM threshold ──────────────────────────────────────────────────────────


def test_prune_over_1000_rows_succeeds(trace_store) -> None:
    """Pruning >1000 rows triggers VACUUM and completes without error."""
    rows = [
        (
            "task-1",
            (datetime.now(UTC) - timedelta(days=100)).isoformat(),
            "intelligence",
            "model",
            10,
            20,
            100.0,
            0.001,
            0,
            "CLOUD",
            "success",
            "",
        )
        for _ in range(1002)
    ]
    trace_store.conn.executemany(
        """INSERT INTO traces
           (task_id, timestamp, expert, model, tokens_in, tokens_out,
            latency_ms, cost_usd, cached, routing, status, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    trace_store.conn.commit()

    deleted = trace_store.prune(max_age_days=90)

    assert deleted == 1002
    remaining = trace_store.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    assert remaining == 0
