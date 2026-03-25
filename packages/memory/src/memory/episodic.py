"""
Episodic memory via Mem0 — extracts and stores long-term facts.

Uses Ollama for the LLM calls required by Mem0's extraction pipeline.
Stores memories locally (no cloud dependency for memory operations).
"""

from __future__ import annotations

import logging

from core.config import get_settings

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Long-term episodic memory using Mem0.

    Automatically extracts facts from conversations:
    - User preferences ("prefers KJV translation")
    - Personal facts ("target salary is $80K")
    - Work patterns ("most productive before 10am")
    - Relationships ("manager's name is X")
    """

    def __init__(self) -> None:
        settings = get_settings()
        try:
            from mem0 import Memory

            self._memory = Memory.from_config(
                {
                    "llm": {
                        "provider": "ollama",
                        "config": {
                            "model": settings.local_llm_model,
                            "ollama_base_url": settings.ollama_host,
                        },
                    },
                    "embedder": {
                        "provider": "ollama",
                        "config": {
                            "model": settings.embedding_model,
                            "ollama_base_url": settings.ollama_host,
                        },
                    },
                    "version": "v1.1",
                }
            )
            self._available = True
            logger.info("Mem0 episodic memory initialized")
        except Exception as e:
            logger.warning(
                "Mem0 initialization failed, episodic memory disabled: %s", e, exc_info=True
            )
            self._available = False
            self._memory = None

    def add(self, text: str, user_id: str = "tremayne", metadata: dict | None = None) -> None:
        """Extract and store facts from a text."""
        if not self._available or not self._memory:
            return
        try:
            self._memory.add(text, user_id=user_id, metadata=metadata or {})
        except Exception as e:
            logger.error("Failed to add memory: %s", e, exc_info=True)

    def search(self, query: str, user_id: str = "tremayne", limit: int = 5) -> list[dict]:
        """Search episodic memories by semantic similarity."""
        if not self._available or not self._memory:
            return []
        try:
            results = self._memory.search(query, user_id=user_id, limit=limit)
            return results.get("results", []) if isinstance(results, dict) else results
        except Exception as e:
            logger.error("Memory search failed: %s", e, exc_info=True)
            return []

    def get_all(self, user_id: str = "tremayne") -> list[dict]:
        """Retrieve all stored memories."""
        if not self._available or not self._memory:
            return []
        try:
            results = self._memory.get_all(user_id=user_id)
            return results.get("results", []) if isinstance(results, dict) else results
        except Exception as e:
            logger.error("Failed to get memories: %s", e, exc_info=True)
            return []
