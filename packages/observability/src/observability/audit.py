"""Agent audit trail — structured event log for dispatch and tool operations.

Records granular events for every agent dispatch, tool call, and system
action. Designed for compliance, debugging, and operational visibility.

Each event includes input/output hashes (not raw data) to enable
correlation without storing sensitive content.

Storage: SQLite WAL mode in the same logs directory as traces.db.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditEventType(StrEnum):
    """Types of auditable events."""

    DISPATCH = "dispatch"
    TOOL_CALL = "tool_call"
    CACHE_HIT = "cache_hit"
    CACHE_STORE = "cache_store"
    ROUTING = "routing"
    FEEDBACK = "feedback"
    MORNING_BRIEF = "morning_brief"
    HEALTH_CHECK = "health_check"
    ERROR = "error"
    MEMORY_WRITE = "memory_write"
    SKILL_EXTRACT = "skill_extract"


def _hash_content(content: str) -> str:
    """SHA-256 hash of content (first 16 chars) for audit correlation."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class AuditStore:
    """SQLite-backed audit event log with WAL mode."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            from core.config import get_settings

            db_path = get_settings().logs_path / "audit.db"

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-4000")  # 4MB
        self._lock = threading.Lock()
        self._create_tables()
        logger.info("audit_store_initialized path=%s", db_path)

    def _create_tables(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    expert TEXT DEFAULT '',
                    task_id TEXT DEFAULT '',
                    model TEXT DEFAULT '',
                    input_hash TEXT DEFAULT '',
                    output_hash TEXT DEFAULT '',
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0.0,
                    cost_usd REAL DEFAULT 0.0,
                    routing TEXT DEFAULT '',
                    status TEXT DEFAULT 'success',
                    metadata TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                    ON audit_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_event_type
                    ON audit_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_audit_expert
                    ON audit_events(expert);
                CREATE INDEX IF NOT EXISTS idx_audit_task_id
                    ON audit_events(task_id);
                CREATE INDEX IF NOT EXISTS idx_audit_status
                    ON audit_events(status);
            """)
            self.conn.commit()

    def record(
        self,
        event_type: AuditEventType | str,
        expert: str = "",
        task_id: str = "",
        model: str = "",
        input_text: str = "",
        output_text: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        routing: str = "",
        status: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an audit event.

        Input/output text is hashed — only the hash is stored, not raw content.
        """
        try:
            with self._lock:
                self.conn.execute(
                    """INSERT INTO audit_events
                       (timestamp, event_type, expert, task_id, model,
                        input_hash, output_hash, tokens_in, tokens_out,
                        latency_ms, cost_usd, routing, status, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(UTC).isoformat(),
                        str(event_type),
                        expert,
                        task_id,
                        model,
                        _hash_content(input_text) if input_text else "",
                        _hash_content(output_text) if output_text else "",
                        tokens_in,
                        tokens_out,
                        latency_ms,
                        cost_usd,
                        routing,
                        status,
                        json.dumps(metadata or {}),
                    ),
                )
                self.conn.commit()
        except sqlite3.Error:
            logger.error("audit_record_failed event=%s", event_type, exc_info=True)

    def get_events(
        self,
        event_type: str | None = None,
        expert: str | None = None,
        hours: int = 24,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit events with optional filters."""
        query = "SELECT * FROM audit_events WHERE timestamp >= datetime('now', ?)"
        params: list[Any] = [f"-{hours} hours"]

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if expert:
            query += " AND expert = ?"
            params.append(expert)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_dispatch_summary(self, hours: int = 24) -> dict[str, Any]:
        """Summarize dispatch activity."""
        rows = self.conn.execute(
            """SELECT
                 expert,
                 COUNT(*) as dispatches,
                 AVG(latency_ms) as avg_latency_ms,
                 SUM(tokens_in) as total_tokens_in,
                 SUM(tokens_out) as total_tokens_out,
                 SUM(cost_usd) as total_cost_usd,
                 SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
               FROM audit_events
               WHERE event_type = 'dispatch'
                 AND timestamp >= datetime('now', ?)
               GROUP BY expert
               ORDER BY dispatches DESC""",
            (f"-{hours} hours",),
        ).fetchall()
        return {"experts": [dict(r) for r in rows]}

    def get_tool_usage(self, hours: int = 24) -> list[dict]:
        """Get tool call frequency and performance."""
        rows = self.conn.execute(
            """SELECT
                 json_extract(metadata, '$.tool_name') as tool_name,
                 expert,
                 COUNT(*) as calls,
                 AVG(latency_ms) as avg_latency_ms,
                 SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
               FROM audit_events
               WHERE event_type = 'tool_call'
                 AND timestamp >= datetime('now', ?)
               GROUP BY tool_name, expert
               ORDER BY calls DESC""",
            (f"-{hours} hours",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_error_events(self, hours: int = 24) -> list[dict]:
        """Get recent error events for debugging."""
        rows = self.conn.execute(
            """SELECT * FROM audit_events
               WHERE status = 'error'
                 AND timestamp >= datetime('now', ?)
               ORDER BY timestamp DESC
               LIMIT 50""",
            (f"-{hours} hours",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_audit_report(self) -> dict[str, Any]:
        """Full daily audit report — all event types summarized."""
        summary = self.conn.execute(
            """SELECT
                 event_type,
                 COUNT(*) as count,
                 SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                 AVG(latency_ms) as avg_latency_ms,
                 SUM(cost_usd) as total_cost_usd
               FROM audit_events
               WHERE timestamp >= datetime('now', '-1 day')
               GROUP BY event_type
               ORDER BY count DESC"""
        ).fetchall()

        total_events = self.conn.execute(
            "SELECT COUNT(*) as c FROM audit_events WHERE timestamp >= datetime('now', '-1 day')"
        ).fetchone()["c"]

        return {
            "total_events": total_events,
            "by_type": [dict(r) for r in summary],
        }

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                logger.debug("audit_store_close_failed", exc_info=True)


# -- Module singleton -----------------------------------------------------------

_store: AuditStore | None = None
_store_lock = threading.Lock()


def get_audit_store(db_path: Path | str | None = None) -> AuditStore:
    """Get or create the module-level AuditStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = AuditStore(db_path)
    return _store
