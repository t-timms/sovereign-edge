"""
Semantic response cache backed by LanceDB.

Cache lookup: embed query → nearest stored query → if cosine similarity
  ≥ CACHE_THRESHOLD, return cached response without LLM call.
Cache store: after each non-cached CLOUD LLM call, store (query, response, vector).

Graceful degradation: all methods silently no-op when LanceDB or Ollama
  is unavailable — the cache is always optional, never blocking.

Threshold 0.92 was chosen empirically: tight enough to avoid false positives
on similar-but-different questions, loose enough to catch rephrased repeats.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from core.config import get_settings

from memory.embeddings import aembed

logger = logging.getLogger(__name__)

CACHE_THRESHOLD = 0.92  # cosine similarity for a cache hit
CACHE_TABLE = "semantic_cache"
MAX_CACHE_AGE_HOURS = 24  # discard stale entries


class SemanticCache:
    """LanceDB-backed semantic response cache."""

    def __init__(self) -> None:
        settings = get_settings()
        self._available = False
        self._db: Any = None
        try:
            import lancedb

            db_path = str(settings.lancedb_path)
            Path(db_path).mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(db_path)
            self._available = True
            logger.info("semantic_cache_initialized db=%s", db_path)
        except Exception as exc:
            logger.warning("semantic_cache_unavailable: %s", exc)

    async def lookup(self, query: str, expert: str = "") -> dict[str, Any] | None:
        """
        Check cache for a semantically similar query.

        Returns a result dict (same shape as LLMGateway.complete()) or None on miss.
        """
        if not self._available or self._db is None:
            return None

        vec = await aembed(query)
        if vec is None:
            return None

        try:
            if CACHE_TABLE not in self._db.table_names():
                return None

            table = self._db.open_table(CACHE_TABLE)
            # metric="cosine" returns cosine *distance* (0=identical, 2=opposite).
            # cosine_similarity = 1 - cosine_distance, giving values in [-1, 1].
            rows = table.search(vec.tolist(), metric="cosine").limit(1).to_list()
            if not rows:
                return None

            row = rows[0]
            dist = float(row.get("_distance", 1.0))
            similarity = max(0.0, 1.0 - dist)

            if similarity < CACHE_THRESHOLD:
                return None

            # Optional expert filter — empty expert matches any
            if expert and row.get("expert", "") not in ("", expert):
                return None

            age_h = (time.time() - float(row.get("timestamp", 0.0))) / 3600
            if age_h > MAX_CACHE_AGE_HOURS:
                return None

            logger.info(
                "cache_hit expert=%s similarity=%.3f age_h=%.1f",
                expert,
                similarity,
                age_h,
            )
            return {
                "content": str(row["response"]),
                "model": "cache",
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
            }

        except Exception:
            logger.debug("cache_lookup_failed", exc_info=True)
            return None

    async def store(self, query: str, response: str, expert: str = "") -> None:
        """Store a query/response pair. Silently skips on any failure."""
        if not self._available or self._db is None:
            return

        vec = await aembed(query)
        if vec is None:
            return

        try:
            record = {
                "query": query,
                "response": response,
                "expert": expert,
                "timestamp": time.time(),
                "vector": vec.tolist(),
            }
            if CACHE_TABLE in self._db.table_names():
                self._db.open_table(CACHE_TABLE).add([record])
            else:
                self._db.create_table(CACHE_TABLE, [record])
            logger.debug("cache_stored expert=%s query_len=%d", expert, len(query))
        except Exception:
            logger.debug("cache_store_failed", exc_info=True)


_instance: SemanticCache | None = None


def get_cache() -> SemanticCache:
    """Module-level singleton."""
    global _instance
    if _instance is None:
        _instance = SemanticCache()
    return _instance
