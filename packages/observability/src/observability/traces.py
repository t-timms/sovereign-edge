"""
SQLite trace store — queryable log of all LLM interactions.

Schema designed for cost tracking, latency analysis, and expert performance monitoring.
WAL mode enabled for concurrent read/write without blocking.
~10MB storage per 10K traces.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime

from core.config import get_settings
from core.types import TaskResult

logger = logging.getLogger(__name__)


class TraceStore:
    """Persistent trace storage in SQLite with WAL mode."""

    def __init__(self) -> None:
        settings = get_settings()
        db_path = settings.logs_path / "traces.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL mode: allows concurrent readers + one writer without blocking
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # safe + fast with WAL
        self.conn.execute("PRAGMA cache_size=-8000")  # 8MB page cache
        # Serialize writes — SQLite WAL allows concurrent reads but not concurrent writes
        self._lock = threading.Lock()
        self._create_tables()
        logger.info("trace_store_initialized path=%s", db_path)

    def _create_tables(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    expert TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0.0,
                    cost_usd REAL DEFAULT 0.0,
                    cached INTEGER DEFAULT 0,
                    routing TEXT DEFAULT 'CLOUD',
                    status TEXT DEFAULT 'success',
                    error_message TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_traces_expert ON traces(expert);
                CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
                CREATE INDEX IF NOT EXISTS idx_traces_model ON traces(model);
                CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);

                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT PRIMARY KEY,
                    total_requests INTEGER DEFAULT 0,
                    total_tokens_in INTEGER DEFAULT 0,
                    total_tokens_out INTEGER DEFAULT 0,
                    total_cost_usd REAL DEFAULT 0.0,
                    cache_hits INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0
                );
            """)
            self.conn.commit()

    def record(self, result: TaskResult, status: str = "success", error: str = "") -> None:
        """Record a completed task result. Safe to call from async context (fast SQLite write)."""
        try:
            with self._lock:
                self.conn.execute(
                    """INSERT INTO traces
                       (task_id, timestamp, expert, model, tokens_in, tokens_out,
                        latency_ms, cost_usd, cached, routing, status, error_message)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(result.task_id),
                        datetime.now(UTC).isoformat(),
                        result.expert,
                        result.model_used,
                        result.tokens_in,
                        result.tokens_out,
                        result.latency_ms,
                        result.cost_usd,
                        1 if result.cached else 0,
                        result.routing,
                        status,
                        error,
                    ),
                )
                self.conn.commit()
        except sqlite3.Error:
            logger.error("trace_record_failed task_id=%s", result.task_id, exc_info=True)

    def get_daily_stats(self, date_str: str | None = None) -> dict:
        """Get aggregated stats for a day (defaults to today UTC)."""
        if date_str is None:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")

        row = self.conn.execute(
            """SELECT
                 COUNT(*) as total_requests,
                 COALESCE(SUM(tokens_in), 0) as total_tokens_in,
                 COALESCE(SUM(tokens_out), 0) as total_tokens_out,
                 COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                 COALESCE(SUM(cached), 0) as cache_hits,
                 COALESCE(SUM(CASE WHEN status='error' THEN 1 ELSE 0 END), 0) as errors,
                 COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
                 GROUP_CONCAT(DISTINCT model) as models_used
               FROM traces
               WHERE date(timestamp) = ?""",
            (date_str,),
        ).fetchone()

        return dict(row) if row else {}

    def get_expert_stats(self, expert: str, days: int = 7) -> list[dict]:
        """Get per-expert stats for the last N days."""
        rows = self.conn.execute(
            """SELECT date(timestamp) as date,
                      COUNT(*) as requests,
                      COALESCE(SUM(tokens_out), 0) as tokens_out,
                      COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
                      COALESCE(SUM(CASE WHEN status='error' THEN 1 ELSE 0 END), 0) as errors
               FROM traces
               WHERE expert = ?
                 AND timestamp >= datetime('now', ?)
               GROUP BY date(timestamp)
               ORDER BY date(timestamp)""",
            (expert, f"-{days} days"),
        ).fetchall()

        return [dict(r) for r in rows]

    def get_model_breakdown(self, days: int = 1) -> list[dict]:
        """Which models are being hit and how often."""
        rows = self.conn.execute(
            """SELECT model,
                      COUNT(*) as requests,
                      COALESCE(SUM(tokens_in + tokens_out), 0) as total_tokens,
                      COALESCE(AVG(latency_ms), 0) as avg_latency_ms
               FROM traces
               WHERE timestamp >= datetime('now', ?)
               GROUP BY model
               ORDER BY requests DESC""",
            (f"-{days} days",),
        ).fetchall()

        return [dict(r) for r in rows]
