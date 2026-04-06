"""
Jina AI web grounding — search (s.jina.ai) and reader (r.jina.ai).

Both endpoints work without a key; supply SE_JINA_API_KEY for higher rate limits.
Free tier: ~200 RPD without key, unlimited with key on free plan.

Production upgrades:
  - Module-level persistent AsyncClient (connection pooling, no SSL handshake per call)
  - 30-minute TTL cache on search results (protects 200 RPD free-tier quota)
  - 2-retry exponential backoff for transient failures
  - aclose() for clean shutdown
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import time
from urllib.parse import quote_plus, urlparse

import httpx
from core.config import get_settings

logger = logging.getLogger(__name__)

_SEARCH_BASE = "https://s.jina.ai"
_READER_BASE = "https://r.jina.ai"
_TIMEOUT = 25.0
_MAX_CHARS = 6_000  # cap to fit LLM context window
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 1.0

# TTL cache for search results — protects free-tier quota
# key → (result_text, monotonic_expiry)
_CACHE_TTL = 1800.0  # 30 minutes
_MAX_CACHE_SIZE = 500  # cap memory usage; LRU eviction when exceeded
_search_cache: dict[str, tuple[str, float]] = {}


# ── Persistent client ──────────────────────────────────────────────────────────
# Lazy-init because Authorization header depends on runtime settings.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        settings = get_settings()
        headers: dict[str, str] = {
            "Accept": "text/markdown",
            "X-Return-Format": "markdown",
        }
        if jina_key := settings.jina_api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {jina_key}"
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers=headers,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _client


# ── Cache helpers ──────────────────────────────────────────────────────────────


def _cache_get(key: str) -> str | None:
    if key in _search_cache:
        value, expiry = _search_cache[key]
        if time.monotonic() < expiry:
            return value
        del _search_cache[key]
    return None


def _cache_set(key: str, value: str) -> None:
    _search_cache[key] = (value, time.monotonic() + _CACHE_TTL)
    # Evict expired entries on every 50th write to prevent unbounded growth.
    if len(_search_cache) % 50 == 0:
        _evict_expired()
    # Hard cap: drop oldest entry if still over limit after TTL eviction
    if len(_search_cache) > _MAX_CACHE_SIZE:
        oldest_key = min(_search_cache, key=lambda k: _search_cache[k][1])
        del _search_cache[oldest_key]
        logger.debug("jina_cache_max_size_eviction remaining=%d", len(_search_cache))


def _evict_expired() -> None:
    now = time.monotonic()
    expired = [k for k, (_, exp) in _search_cache.items() if now >= exp]
    for k in expired:
        del _search_cache[k]
    if expired:
        logger.debug("jina_cache_evicted count=%d remaining=%d", len(expired), len(_search_cache))


# Private IP ranges — SSRF guard for fetch()
_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


_DNS_TIMEOUT = 5.0  # seconds — prevents hung thread on slow/malicious DNS


async def _is_private_url(url: str) -> bool:
    """Return True if the URL resolves to a private/loopback address.

    DNS resolution runs in a thread-pool executor so the async event loop
    is never blocked by a slow or hanging DNS query. A 5-second timeout
    prevents indefinite hangs from unresponsive resolvers.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return True
        loop = asyncio.get_running_loop()
        resolved = await asyncio.wait_for(
            loop.run_in_executor(None, socket.gethostbyname, hostname),
            timeout=_DNS_TIMEOUT,
        )
        addr = ipaddress.ip_address(resolved)
        return any(addr in net for net in _PRIVATE_NETS)
    except (TimeoutError, OSError, ValueError):
        return True  # Fail closed on any resolution error or timeout


# ── Public API ─────────────────────────────────────────────────────────────────


async def search(query: str, max_results: int = 5) -> str:
    """Search the live web via Jina and return clean markdown results.

    Results are cached for 30 minutes to protect the 200 RPD free-tier quota.
    """
    cache_key = f"{query}|{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("jina_search_cache_hit query=%r", query)
        return cached

    url = f"{_SEARCH_BASE}/{quote_plus(query)}"
    params = {"num_results": str(max_results)}
    client = _get_client()

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            result = resp.text[:_MAX_CHARS]
            _cache_set(cache_key, result)
            logger.debug("jina_search_ok query=%r chars=%d", query, len(result))
            return result
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "jina_search_http_error status=%s query=%r attempt=%d",
                exc.response.status_code,
                query,
                attempt + 1,
            )
            if exc.response.status_code < 500:
                return ""  # 4xx — don't retry
        except httpx.HTTPError as exc:
            logger.warning(
                "jina_search_failed query=%r attempt=%d error=%s",
                query,
                attempt + 1,
                exc,
            )
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))

    return ""


async def fetch(url: str) -> str:
    """Fetch and clean any URL as markdown via Jina Reader.

    No cache — URLs are typically unique per call.
    URLs resolving to private/loopback addresses are rejected (SSRF guard).
    """
    if await _is_private_url(url):
        logger.warning("jina_fetch_blocked_ssrf url=%r", url)
        return ""
    reader_url = f"{_READER_BASE}/{url}"
    client = _get_client()

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get(reader_url)
            resp.raise_for_status()
            result = resp.text[:_MAX_CHARS]
            logger.debug("jina_fetch_ok url=%r chars=%d", url, len(result))
            return result
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "jina_fetch_http_error status=%s url=%r attempt=%d",
                exc.response.status_code,
                url,
                attempt + 1,
            )
            if exc.response.status_code < 500:
                return ""  # 4xx — don't retry
        except httpx.HTTPError as exc:
            logger.warning(
                "jina_fetch_failed url=%r attempt=%d error=%s",
                url,
                attempt + 1,
                exc,
            )
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))

    return ""


async def aclose() -> None:
    """Close the persistent HTTP client. Call during application shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
