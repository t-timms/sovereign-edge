from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from core.config import Settings
from core.exceptions import MemoryError as SovereignMemoryError
from memory.store import MemoryEntry, MemoryStore


@pytest.fixture()
def settings(tmp_path: pytest.fixture) -> Settings:
    return Settings(lancedb_path=tmp_path / "lancedb", mem0_user_id="test_user")


# ── MemoryEntry ───────────────────────────────────────────────────────────────


def test_memory_entry_fields() -> None:
    e = MemoryEntry(text="prayer is powerful", score=0.9, metadata={"source": "test"})
    assert e.text == "prayer is powerful"
    assert e.score == 0.9
    assert e.metadata["source"] == "test"


# ── add_memory ────────────────────────────────────────────────────────────────


def test_add_memory_calls_mem0(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_mem0 = MagicMock()
    store._mem0 = mock_mem0

    store.add_memory("I enjoy reading Psalms", user_id="john")

    mock_mem0.add.assert_called_once_with("I enjoy reading Psalms", user_id="john")


def test_add_memory_uses_settings_user_id(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_mem0 = MagicMock()
    store._mem0 = mock_mem0

    store.add_memory("some memory text")

    mock_mem0.add.assert_called_once_with("some memory text", user_id="test_user")


def test_add_memory_raises_on_mem0_failure(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_mem0 = MagicMock()
    mock_mem0.add.side_effect = RuntimeError("mem0 down")
    store._mem0 = mock_mem0

    with pytest.raises(SovereignMemoryError, match="Failed to add memory"):
        store.add_memory("text")


# ── search_memory ─────────────────────────────────────────────────────────────


def test_search_memory_returns_entries(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_mem0 = MagicMock()
    mock_mem0.search.return_value = {
        "results": [
            {"memory": "I like Python", "score": 0.85, "metadata": {}},
            {"memory": "I work in Dallas", "score": 0.72, "metadata": {}},
        ]
    }
    store._mem0 = mock_mem0

    results = store.search_memory("Python career", limit=2)

    assert len(results) == 2
    assert all(isinstance(r, MemoryEntry) for r in results)
    assert results[0].text == "I like Python"
    assert results[0].score == 0.85


def test_search_memory_returns_empty_list_on_failure(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_mem0 = MagicMock()
    mock_mem0.search.side_effect = RuntimeError("timeout")
    store._mem0 = mock_mem0

    results = store.search_memory("anything")

    assert results == []


def test_search_memory_empty_results(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_mem0 = MagicMock()
    mock_mem0.search.return_value = {"results": []}
    store._mem0 = mock_mem0

    results = store.search_memory("query with no matches")

    assert results == []


# ── format_context ────────────────────────────────────────────────────────────


def test_format_context_returns_formatted_string(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_mem0 = MagicMock()
    mock_mem0.search.return_value = {
        "results": [{"memory": "User prefers Python", "score": 0.9, "metadata": {}}]
    }
    store._mem0 = mock_mem0

    ctx = store.format_context("programming")

    assert "Relevant context from memory" in ctx
    assert "User prefers Python" in ctx


def test_format_context_returns_empty_when_no_memories(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_mem0 = MagicMock()
    mock_mem0.search.return_value = {"results": []}
    store._mem0 = mock_mem0

    ctx = store.format_context("no match query")

    assert ctx == ""


# ── _get_mem0 lazy init ───────────────────────────────────────────────────────


def test_get_mem0_raises_on_init_failure(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    with patch("memory.store.Memory", side_effect=RuntimeError("no config")):
        with pytest.raises(SovereignMemoryError, match="Failed to initialize Mem0"):
            store._get_mem0()


# ── LanceDB upsert / search ───────────────────────────────────────────────────


def test_upsert_chunks_creates_table(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_db = MagicMock()
    mock_db.table_names.return_value = []
    store._lance = mock_db

    chunks: list[dict[str, Any]] = [{"text": "chunk1", "vector": [0.1, 0.2]}]
    store.upsert_chunks("bible", chunks)

    mock_db.create_table.assert_called_once_with("bible", data=chunks)


def test_upsert_chunks_adds_to_existing_table(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_db.table_names.return_value = ["bible"]
    mock_db.open_table.return_value = mock_table
    store._lance = mock_db

    chunks: list[dict[str, Any]] = [{"text": "chunk2", "vector": [0.3, 0.4]}]
    store.upsert_chunks("bible", chunks)

    mock_table.add.assert_called_once_with(chunks)


def test_upsert_chunks_raises_on_failure(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_db = MagicMock()
    mock_db.table_names.return_value = []
    mock_db.create_table.side_effect = RuntimeError("disk full")
    store._lance = mock_db

    with pytest.raises(SovereignMemoryError, match="Failed to upsert chunks"):
        store.upsert_chunks("bible", [{"text": "x", "vector": [0.1]}])


def test_vector_search_returns_results(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_table.search.return_value.limit.return_value.to_list.return_value = [
        {"text": "In the beginning", "vector": [0.1, 0.2], "_distance": 0.1}
    ]
    mock_db.table_names.return_value = ["bible"]
    mock_db.open_table.return_value = mock_table
    store._lance = mock_db

    results = store.vector_search("bible", [0.1, 0.2], limit=3)

    assert len(results) == 1
    assert results[0]["text"] == "In the beginning"


def test_vector_search_returns_empty_for_missing_table(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_db = MagicMock()
    mock_db.table_names.return_value = []
    store._lance = mock_db

    results = store.vector_search("nonexistent_table", [0.1, 0.2])

    assert results == []


def test_vector_search_returns_empty_on_exception(settings: Settings) -> None:
    store = MemoryStore(settings=settings)
    mock_db = MagicMock()
    mock_db.table_names.return_value = ["bible"]
    mock_db.open_table.side_effect = RuntimeError("corrupted")
    store._lance = mock_db

    results = store.vector_search("bible", [0.1, 0.2])

    assert results == []
