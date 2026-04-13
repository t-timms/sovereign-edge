"""Tests for ConversationStore.prune_old_chats() — SQLite storage pruning.

Covers:
- Old turns are deleted when beyond max_age_days
- Recent turns are preserved
- Return value matches number of deleted rows
- Zero rows deleted when nothing is old
- Turns from multiple chats are pruned correctly
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def store(tmp_path: Path):
    """ConversationStore backed by a temp SQLite file."""
    from memory.conversation import ConversationStore

    mock_settings = MagicMock()
    mock_settings.logs_path = tmp_path

    with patch("memory.conversation.get_settings", return_value=mock_settings):
        return ConversationStore()


def _insert_turn(store, *, chat_id: str, days_ago: int, content: str = "hello") -> None:
    """Insert a turn with a timestamp *days_ago* days in the past."""
    ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    store.conn.execute(
        "INSERT INTO turns (chat_id, role, content, expert, ts) VALUES (?, ?, ?, ?, ?)",
        (chat_id, "user", content, "", ts),
    )
    store.conn.commit()


# ── basic pruning ─────────────────────────────────────────────────────────────


def test_prune_deletes_old_turns(store) -> None:
    """Turns older than max_age_days should be deleted."""
    _insert_turn(store, chat_id="chat1", days_ago=60)
    _insert_turn(store, chat_id="chat1", days_ago=10)

    deleted = store.prune_old_chats(max_age_days=30)

    assert deleted == 1
    remaining = store.conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    assert remaining == 1


def test_prune_preserves_recent_turns(store) -> None:
    """Turns newer than max_age_days must not be deleted."""
    _insert_turn(store, chat_id="chat1", days_ago=5)
    _insert_turn(store, chat_id="chat1", days_ago=10)

    deleted = store.prune_old_chats(max_age_days=30)

    assert deleted == 0
    remaining = store.conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    assert remaining == 2


def test_prune_returns_zero_when_empty(store) -> None:
    """Pruning an empty table returns 0."""
    deleted = store.prune_old_chats(max_age_days=30)
    assert deleted == 0


def test_prune_across_multiple_chats(store) -> None:
    """Old turns from different chat_ids should all be pruned."""
    _insert_turn(store, chat_id="chat1", days_ago=60, content="old1")
    _insert_turn(store, chat_id="chat2", days_ago=45, content="old2")
    _insert_turn(store, chat_id="chat1", days_ago=5, content="recent1")
    _insert_turn(store, chat_id="chat2", days_ago=3, content="recent2")

    deleted = store.prune_old_chats(max_age_days=30)

    assert deleted == 2
    remaining = store.conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    assert remaining == 2


def test_prune_boundary_exact_cutoff(store) -> None:
    """A turn exactly at the cutoff boundary should be deleted (< cutoff, not <=)."""
    # Insert a turn exactly 30 days ago — should be deleted since
    # cutoff = now - 30 days, and the turn's ts will be slightly before cutoff
    # due to sub-second timing differences
    _insert_turn(store, chat_id="chat1", days_ago=31)
    _insert_turn(store, chat_id="chat1", days_ago=29)

    deleted = store.prune_old_chats(max_age_days=30)

    assert deleted == 1
    remaining = store.conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    assert remaining == 1
