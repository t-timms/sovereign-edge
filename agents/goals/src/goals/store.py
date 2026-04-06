"""
GoalStore — WAL-mode SQLite persistence for personal goals.

Thread-safe via threading.Lock. Default path: settings.ssd_root/goals.db.
Follows the same pattern as TraceStore in packages/observability.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS goals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    description  TEXT    NOT NULL DEFAULT '',
    target_date  TEXT,
    progress_pct INTEGER NOT NULL DEFAULT 0 CHECK (progress_pct BETWEEN 0 AND 100),
    status       TEXT    NOT NULL DEFAULT 'active'
                         CHECK (status IN ('active', 'complete', 'paused')),
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals (status);
CREATE INDEX IF NOT EXISTS idx_goals_target ON goals (target_date);
"""


@dataclass
class GoalRecord:
    """A single goal returned from the store."""

    id: int
    title: str
    description: str
    target_date: str | None
    progress_pct: int
    status: str
    created_at: str
    updated_at: str

    @property
    def days_remaining(self) -> int | None:
        """Days until target_date, or None if no date set."""
        if not self.target_date:
            return None
        try:
            target = datetime.fromisoformat(self.target_date).replace(tzinfo=UTC)
            delta = target - datetime.now(UTC)
            return max(0, delta.days)
        except ValueError:
            return None


class GoalStore:
    """Thread-safe SQLite-backed goal persistence."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            from core.config import get_settings

            s = get_settings()
            db_path = s.goals_db_path or (s.ssd_root / "goals.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Write operations ──────────────────────────────────────────────────────

    def add_goal(
        self,
        title: str,
        description: str = "",
        target_date: str | None = None,
    ) -> int:
        """Insert a new active goal. Returns the new goal id."""
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO goals (title, description, target_date, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (title.strip(), description.strip(), target_date, now, now),
            )
            goal_id = cur.lastrowid
        logger.info("goal_added id=%d title=%r", goal_id, title)
        return goal_id  # type: ignore[return-value]

    def update_progress(self, goal_id: int, pct: int) -> None:
        """Set progress percentage (clamped to 0-100)."""
        clamped = max(0, min(100, pct))
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE goals SET progress_pct=?, updated_at=? WHERE id=?",
                (clamped, now, goal_id),
            )
        logger.info("goal_progress_updated id=%d pct=%d", goal_id, clamped)

    def mark_complete(self, goal_id: int) -> None:
        """Set status to 'complete' and progress to 100."""
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE goals SET status='complete', progress_pct=100, updated_at=? WHERE id=?",
                (now, goal_id),
            )
        logger.info("goal_completed id=%d", goal_id)

    def pause_goal(self, goal_id: int) -> None:
        """Set status to 'paused'."""
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE goals SET status='paused', updated_at=? WHERE id=?",
                (now, goal_id),
            )

    # ── Read operations ───────────────────────────────────────────────────────

    def list_goals(self, status: str = "active") -> list[GoalRecord]:
        """Return all goals with the given status, ordered by target_date."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM goals WHERE status=? ORDER BY target_date ASC NULLS LAST, id ASC",
                (status,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_urgent(self, limit: int = 3) -> list[GoalRecord]:
        """Return the most urgent active goals for the morning brief.

        Urgency: goals with a target_date first (soonest first),
        then goals without a date (ordered by creation).
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM goals
                WHERE status = 'active'
                ORDER BY
                    CASE WHEN target_date IS NULL THEN 1 ELSE 0 END,
                    target_date ASC,
                    id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_by_id(self, goal_id: int) -> GoalRecord | None:
        """Fetch a single goal by id, or None if not found."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        return _row_to_record(row) if row else None


def _row_to_record(row: sqlite3.Row) -> GoalRecord:
    return GoalRecord(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        target_date=row["target_date"],
        progress_pct=row["progress_pct"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
