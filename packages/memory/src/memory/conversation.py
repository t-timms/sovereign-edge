"""
Per-chat conversation history in SQLite.

Stores the last MAX_STORED turns per chat_id and exposes the most recent
MAX_TURNS for injection into expert prompts as proper message turns.
Same WAL-mode SQLite path as traces — co-located for easy backup.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime

from core.config import get_settings

logger = logging.getLogger(__name__)

MAX_TURNS = 8  # turns injected into each expert request
MAX_STORED = 40  # turns retained per chat_id before pruning
MAX_HISTORY_TOKENS = 2000  # per-request token budget for history (4 chars ≈ 1 token)


class ConversationStore:
    """SQLite-backed per-chat message history with automatic pruning."""

    def __init__(self) -> None:
        settings = get_settings()
        db_path = settings.logs_path / "conversations.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-4000")
        # Serialize writes — SQLite WAL allows concurrent reads but not concurrent writes
        self._lock = threading.Lock()
        self._create_tables()
        logger.info("conversation_store_initialized path=%s", db_path)

    def _create_tables(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS turns (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id  TEXT NOT NULL,
                    role     TEXT NOT NULL,
                    content  TEXT NOT NULL,
                    expert    TEXT DEFAULT '',
                    ts       TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_chat
                    ON turns(chat_id, id DESC);
            """)
            self.conn.commit()

    def add_turn(
        self,
        chat_id: str,
        role: str,
        content: str,
        expert: str = "",
    ) -> None:
        """Append a turn and prune if over MAX_STORED."""
        if not chat_id or not chat_id.strip():
            logger.warning("conversation_add_turn_skipped — empty chat_id")
            return
        if role not in ("user", "assistant", "system"):
            logger.warning("conversation_add_turn_skipped — invalid role=%r", role)
            return
        try:
            with self._lock:
                self.conn.execute(
                    "INSERT INTO turns (chat_id, role, content, expert, ts) VALUES (?,?,?,?,?)",
                    (chat_id, role, content, expert, datetime.now(UTC).isoformat()),
                )
                # Keep only the newest MAX_STORED turns per chat
                self.conn.execute(
                    """DELETE FROM turns
                       WHERE chat_id = ? AND id NOT IN (
                           SELECT id FROM turns WHERE chat_id = ?
                           ORDER BY id DESC LIMIT ?)""",
                    (chat_id, chat_id, MAX_STORED),
                )
                self.conn.commit()
        except sqlite3.Error:
            logger.error("conversation_add_turn_failed chat_id=%s", chat_id, exc_info=True)

    def get_recent(self, chat_id: str, n: int = MAX_TURNS) -> list[dict[str, str]]:
        """Return the last *n* turns as [{"role":..., "content":...}] oldest-first.

        Trims oldest turns until estimated token count is within MAX_HISTORY_TOKENS.
        This caps context cost regardless of message length.
        """
        try:
            rows = self.conn.execute(
                "SELECT role, content FROM turns WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (chat_id, n),
            ).fetchall()
            turns = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        except sqlite3.Error:
            logger.error("conversation_get_recent_failed chat_id=%s", chat_id, exc_info=True)
            return []

        # Trim oldest turns while over budget — always keep the most recent turn
        while len(turns) > 1 and sum(len(t["content"]) for t in turns) // 4 > MAX_HISTORY_TOKENS:
            turns.pop(0)

        return turns

    def get_recent_json(self, chat_id: str, n: int = MAX_TURNS) -> str:
        """JSON-encoded history string for TaskRequest.context injection."""
        turns = self.get_recent(chat_id, n)
        return json.dumps(turns) if turns else ""

    def clear(self, chat_id: str) -> None:
        """Delete all stored turns for a chat_id."""
        try:
            with self._lock:
                self.conn.execute("DELETE FROM turns WHERE chat_id=?", (chat_id,))
                self.conn.commit()
        except sqlite3.Error:
            logger.error("conversation_clear_failed chat_id=%s", chat_id, exc_info=True)

    def prune_old_chats(self, max_age_days: int = 30) -> int:
        """Delete turns older than *max_age_days* and return the count of deleted rows."""
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat()
        try:
            with self._lock:
                cursor = self.conn.execute("DELETE FROM turns WHERE ts < ?", (cutoff,))
                deleted = cursor.rowcount
                self.conn.commit()
                logger.info(
                    "conversation_store_pruned deleted=%d max_age_days=%d",
                    deleted,
                    max_age_days,
                )
                return deleted
        except sqlite3.Error:
            logger.error("conversation_store_prune_failed", exc_info=True)
            return 0

    def close(self) -> None:
        """Close the underlying SQLite connection. Safe to call multiple times."""
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                logger.debug("conversation_store_close_failed", exc_info=True)


_instance: ConversationStore | None = None
_instance_lock = threading.Lock()


def get_conversation_store() -> ConversationStore:
    """Thread-safe module-level singleton — preserves connection across calls."""
    global _instance
    if _instance is None:
        with _instance_lock:
            # Double-checked locking: re-check under lock to prevent double init
            if _instance is None:
                _instance = ConversationStore()
    return _instance
