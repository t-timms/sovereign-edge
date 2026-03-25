"""
SQLite trace store — queryable log of all LLM interactions.

Schema designed for cost tracking, latency analysis, and squad performance monitoring.
~10MB storage per 10K traces with rotation.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from core.config import get_settings
from core.types import TaskResult

logger = logging.getLogger(__name__)


class TraceStore:
    """Persistent trace storage in SQLite."""

    def __init__(self) -> None:
        settings = get_settings()
        db_path = settings.logs_path / "traces.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("Trace store initialized at %s", db_path)

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                squad TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0.0,
                cost_usd REAL DEFAULT 0.0,
                cached INTEGER DEFAULT 0,
                status TEXT DEFAULT 'success',
                error_message TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_traces_squad ON traces(squad);
            CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
            CREATE INDEX IF NOT EXISTS idx_traces_model ON traces(model);

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
        """Record a completed task result."""
        self.conn.execute(
            """INSERT INTO traces
               (task_id, timestamp, squad, model, tokens_in, tokens_out,
                latency_ms, cost_usd, cached, status, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(result.task_id),
                datetime.utcnow().isoformat(),
                result.squad,
                result.model_used,
                result.tokens_in,
                result.tokens_out,
                result.latency_ms,
                result.cost_usd,
                1 if result.cached else 0,
                status,
                error,
            ),
        )
        self.conn.commit()

    def get_daily_stats(self, date_str: str | None = None) -> dict:
        """Get aggregated stats for a day."""
        if date_str is None:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")

        row = self.conn.execute(
            """SELECT
                 COUNT(*) as total_requests,
                 COALESCE(SUM(tokens_in), 0) as total_tokens_in,
                 COALESCE(SUM(tokens_out), 0) as total_tokens_out,
                 COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                 COALESCE(SUM(cached), 0) as cache_hits,
                 COALESCE(SUM(CASE WHEN status='error' THEN 1 ELSE 0 END), 0) as errors
               FROM traces
               WHERE date(timestamp) = ?""",
            (date_str,),
        ).fetchone()

        return dict(row) if row else {}

    def get_squad_stats(self, squad: str, days: int = 7) -> list[dict]:
        """Get per-squad stats for the last N days."""
        rows = self.conn.execute(
            """SELECT date(timestamp) as date,
                      COUNT(*) as requests,
                      COALESCE(SUM(tokens_out), 0) as tokens,
                      COALESCE(AVG(latency_ms), 0) as avg_latency_ms
               FROM traces
               WHERE squad = ?
                 AND timestamp >= datetime('now', ?)
               GROUP BY date(timestamp)
               ORDER BY date(timestamp)""",
            (squad, f"-{days} days"),
        ).fetchall()

        return [dict(r) for r in rows]
