"""
LanceDB vector store — embedded, disk-based, ~100MB RAM.

Provides domain-specific tables for each expert plus a shared personal table.
Uses Qwen3-Embedding-0.6B via Ollama for embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lancedb
from core.config import get_settings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VectorRecord(BaseModel):
    """Base schema for vector records."""

    text: str
    source: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class VectorStore:
    """
    Multi-table vector store backed by LanceDB.

    Tables:
      - bible: STEPBible verses, cross-references, commentary
      - career: Job listings, resume versions, match history
      - research: arXiv papers, AI news, HF daily papers
      - content: Generated posts, scripts, video metadata
      - personal: Journal entries, preferences, life context
    """

    TABLES: list[str] = ["bible", "career", "research", "content", "personal"]  # noqa: RUF012

    def __init__(self) -> None:
        settings = get_settings()
        self.db_path = str(settings.lancedb_path)
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        logger.info("LanceDB connected at %s", self.db_path)

    def add(self, table_name: str, records: list[dict[str, Any]]) -> int:
        """
        Add records to a table. Creates table if it doesn't exist.

        Each record must have at least a 'text' field.
        Embeddings are generated automatically by LanceDB using the configured model.

        Returns number of records added.
        """
        if table_name not in self.TABLES:
            raise ValueError(f"Unknown table: {table_name}. Valid: {self.TABLES}")

        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            table.add(records)
        else:
            self.db.create_table(table_name, records)

        logger.info("Added %d records to %s", len(records), table_name)
        return len(records)

    def search(
        self,
        table_name: str,
        query: str,
        limit: int = 5,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search in a table.

        Returns list of dicts with 'text', 'source', 'metadata', '_distance'.
        """
        if table_name not in self.db.table_names():
            logger.warning("Table %s does not exist yet", table_name)
            return []

        table = self.db.open_table(table_name)
        query_builder = table.search(query).limit(limit)

        if filter_expr:
            query_builder = query_builder.where(filter_expr)

        results = query_builder.to_list()
        return results  # type: ignore[return-value]

    def hybrid_search(
        self,
        table_name: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search: semantic + BM25 keyword matching.
        Requires FTS index on the table (created via create_fts_index).
        """
        if table_name not in self.db.table_names():
            return []

        table = self.db.open_table(table_name)
        try:
            results = table.search(query, query_type="hybrid").limit(limit).to_list()
            return results  # type: ignore[return-value]
        except Exception:
            # Fall back to pure semantic if FTS index doesn't exist
            return self.search(table_name, query, limit)

    def count(self, table_name: str) -> int:
        """Return number of records in a table."""
        if table_name not in self.db.table_names():
            return 0
        table = self.db.open_table(table_name)
        return table.count_rows()  # type: ignore[return-value]
