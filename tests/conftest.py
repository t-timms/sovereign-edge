"""Reset module-level singletons and caches between tests.

Every test that imports llm.gateway, search.*, etc. gets a clean slate.
autouse=True means no decorator needed on individual tests.
"""

from __future__ import annotations

import sys
from datetime import date

import pytest


@pytest.fixture(autouse=True)
def reset_module_singletons() -> None:
    """Tear down all module-level state after each test."""
    yield
    _reset()


def _reset() -> None:
    # LLM gateway singleton
    if "llm.gateway" in sys.modules:
        import llm.gateway as gw  # type: ignore[import]

        gw._instance = None

    # HTTP clients for all search modules
    for mod_name in ("search.jina", "search.arxiv", "search.bible", "search.hf"):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            mod._client = None  # type: ignore[attr-defined]

    # Jina TTL cache
    if "search.jina" in sys.modules:
        import search.jina as jina  # type: ignore[import]

        jina._search_cache.clear()

    # arXiv daily novelty filter
    if "search.arxiv" in sys.modules:
        import search.arxiv as arxiv  # type: ignore[import]

        arxiv._seen_ids = set()
        arxiv._seen_date = date.today()
