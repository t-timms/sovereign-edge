"""Tests for search package modules — all external HTTP calls are mocked."""

from __future__ import annotations

import time
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mock_response(
    text: str = "",
    status_code: int = 200,
    json_data: object = None,
) -> MagicMock:
    """Build a fake httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json = MagicMock(return_value=json_data or {})
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


def _make_async_client(response: MagicMock) -> MagicMock:
    """Return a mock AsyncClient whose .get() returns *response*."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Jina
# ─────────────────────────────────────────────────────────────────────────────


class TestJinaSearch:
    async def test_cache_miss_fetches_and_caches(self) -> None:
        import search.jina as jina

        expected = "## Result\nSome content"
        client = _make_async_client(_mock_response(text=expected))
        jina._client = client

        result = await jina.search("pytorch optimizer", max_results=3)

        assert result == expected
        client.get.assert_awaited_once()
        # Second call must be a cache hit (get not called again)
        result2 = await jina.search("pytorch optimizer", max_results=3)
        assert result2 == expected
        client.get.assert_awaited_once()  # still once — served from cache

    async def test_cache_hit_skips_http(self) -> None:
        import search.jina as jina

        # Pre-populate cache with a future expiry
        cache_key = "warm cache|5"
        jina._search_cache[cache_key] = ("cached content", time.monotonic() + 600)

        client = _make_async_client(_mock_response(text="should not reach"))
        jina._client = client

        result = await jina.search("warm cache", max_results=5)

        assert result == "cached content"
        client.get.assert_not_awaited()

    async def test_expired_cache_refetches(self) -> None:
        import search.jina as jina

        cache_key = "stale|5"
        jina._search_cache[cache_key] = ("old content", time.monotonic() - 1)

        fresh = "fresh content"
        client = _make_async_client(_mock_response(text=fresh))
        jina._client = client

        result = await jina.search("stale", max_results=5)

        assert result == fresh
        client.get.assert_awaited_once()

    async def test_http_error_returns_empty_string(self) -> None:
        import httpx
        import search.jina as jina

        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("network failure"))
        jina._client = client

        result = await jina.search("broken query")
        assert result == ""

    async def test_4xx_does_not_retry(self) -> None:
        """4xx responses should return '' immediately without retrying."""
        import search.jina as jina

        resp = _mock_response(status_code=429)
        client = _make_async_client(resp)
        jina._client = client

        result = await jina.search("rate limited")
        # Called exactly once — no retries for 4xx
        assert client.get.await_count == 1
        assert result == ""

    async def test_result_truncated_to_max_chars(self) -> None:
        import search.jina as jina

        long_text = "x" * 10_000
        client = _make_async_client(_mock_response(text=long_text))
        jina._client = client

        result = await jina.search("long result")
        assert len(result) <= jina._MAX_CHARS


class TestJinaFetch:
    async def test_fetch_returns_content(self) -> None:
        import search.jina as jina

        content = "# Page Title\nSome body text"
        client = _make_async_client(_mock_response(text=content))
        jina._client = client

        result = await jina.fetch("https://example.com/article")
        assert result == content

    async def test_fetch_http_error_returns_empty_string(self) -> None:
        import httpx
        import search.jina as jina

        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("timeout"))
        jina._client = client

        result = await jina.fetch("https://broken.example.com")
        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# arXiv
# ─────────────────────────────────────────────────────────────────────────────

_ATOM_TEMPLATE = dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/{arxiv_id}</id>
        <title>{title}</title>
        <summary>{summary}</summary>
        <author><name>Alice Smith</name></author>
      </entry>
    </feed>
""")


class TestArxivFetch:
    async def test_parses_paper_fields(self) -> None:
        import search.arxiv as arxiv

        xml = _ATOM_TEMPLATE.format(
            arxiv_id="2503.00001",
            title="Advances in GRPO Training",
            summary="This paper describes a new GRPO method.",
        )
        client = _make_async_client(_mock_response(text=xml))
        arxiv._client = client

        papers = await arxiv.fetch_recent(max_results=1)

        assert len(papers) == 1
        assert papers[0]["title"] == "Advances in GRPO Training"
        assert "GRPO" in papers[0]["summary"]
        assert papers[0]["url"] == "https://arxiv.org/abs/2503.00001"
        assert papers[0]["authors"] == "Alice Smith"

    async def test_novelty_filter_deduplicates(self) -> None:
        """Papers returned in call 1 must not appear in call 2 (same day)."""
        import search.arxiv as arxiv

        xml = _ATOM_TEMPLATE.format(
            arxiv_id="2503.99999",
            title="Duplicate Paper",
            summary="Will be filtered on second call.",
        )
        client = _make_async_client(_mock_response(text=xml))
        arxiv._client = client

        papers1 = await arxiv.fetch_recent(max_results=5)
        assert len(papers1) == 1

        papers2 = await arxiv.fetch_recent(max_results=5)
        assert len(papers2) == 0  # already seen today

    async def test_empty_feed_returns_empty_list(self) -> None:
        import search.arxiv as arxiv

        empty_xml = dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom"></feed>
        """)
        client = _make_async_client(_mock_response(text=empty_xml))
        arxiv._client = client

        papers = await arxiv.fetch_recent(max_results=5)
        assert papers == []

    async def test_malformed_xml_returns_empty_list(self) -> None:
        import search.arxiv as arxiv

        client = _make_async_client(_mock_response(text="<<<not xml>>>"))
        arxiv._client = client

        papers = await arxiv.fetch_recent(max_results=5)
        assert papers == []

    async def test_http_error_returns_empty_list(self) -> None:
        import httpx
        import search.arxiv as arxiv

        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("server down"))
        arxiv._client = client

        papers = await arxiv.fetch_recent(max_results=5)
        assert papers == []

    def test_format_papers_renders_markdown(self) -> None:
        from search.arxiv import format_papers

        papers = [
            {
                "title": "Test Paper",
                "authors": "Bob Jones",
                "summary": "A summary.",
                "url": "https://arxiv.org/abs/1234",
            },
        ]
        text = format_papers(papers)
        assert "Test Paper" in text
        assert "Bob Jones" in text
        assert "https://arxiv.org/abs/1234" in text

    def test_format_papers_empty_returns_empty_string(self) -> None:
        from search.arxiv import format_papers

        assert format_papers([]) == ""


# ─────────────────────────────────────────────────────────────────────────────
# Bible
# ─────────────────────────────────────────────────────────────────────────────


class TestBible:
    async def test_random_verse_returns_dict(self) -> None:
        import search.bible as bible

        json_data = {"reference": "John 3:16", "text": "For God so loved the world"}
        client = _make_async_client(_mock_response(json_data=json_data))
        bible._client = client

        verse = await bible.random_verse()

        assert verse["reference"] == "John 3:16"
        assert "God" in verse["text"]
        assert verse["translation"] == "KJV"

    async def test_random_verse_http_error_returns_empty(self) -> None:
        import httpx
        import search.bible as bible

        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("network error"))
        bible._client = client

        verse = await bible.random_verse()
        assert verse == {"reference": "", "text": "", "translation": ""}

    async def test_lookup_returns_assembled_text(self) -> None:
        import search.bible as bible

        json_data = {
            "reference": "Psalm 23:1",
            "verses": [{"text": "The LORD is my shepherd; "}, {"text": "I shall not want."}],
        }
        client = _make_async_client(_mock_response(json_data=json_data))
        bible._client = client

        verse = await bible.lookup("Psalm 23:1")

        assert verse["reference"] == "Psalm 23:1"
        assert "LORD" in verse["text"]
        assert "want" in verse["text"]

    async def test_lookup_http_error_returns_reference_with_empty_text(self) -> None:
        import httpx
        import search.bible as bible

        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("timeout"))
        bible._client = client

        verse = await bible.lookup("Genesis 1:1")
        assert verse["reference"] == "Genesis 1:1"
        assert verse["text"] == ""

    def test_format_verse_non_empty(self) -> None:
        from search.bible import format_verse

        verse = {
            "reference": "John 3:16",
            "text": "For God so loved the world",
            "translation": "KJV",
        }
        result = format_verse(verse)
        assert "John 3:16" in result
        assert "KJV" in result
        assert "God" in result

    def test_format_verse_empty_text_returns_empty_string(self) -> None:
        from search.bible import format_verse

        verse = {"reference": "John 3:16", "text": "", "translation": "KJV"}
        assert format_verse(verse) == ""


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face Daily Papers
# ─────────────────────────────────────────────────────────────────────────────

_HF_PAPERS_PAYLOAD = [
    {
        "paper": {
            "id": "2503.12345",
            "title": "FlashAttention-4",
            "summary": "A faster attention mechanism for large models.",
            "upvotes": 42,
        },
        "numComments": 5,  # must NOT be used for upvotes
    }
]


class TestHFPapers:
    async def test_parses_title_summary_url(self) -> None:
        import search.hf as hf

        client = _make_async_client(_mock_response(json_data=_HF_PAPERS_PAYLOAD))
        hf._client = client

        papers = await hf.fetch_daily_papers()

        assert len(papers) == 1
        assert papers[0]["title"] == "FlashAttention-4"
        assert "faster" in papers[0]["summary"]
        assert papers[0]["url"] == "https://huggingface.co/papers/2503.12345"

    async def test_upvotes_field_not_num_comments(self) -> None:
        """Critical regression: upvotes must come from paper.upvotes, not item.numComments."""
        import search.hf as hf

        client = _make_async_client(_mock_response(json_data=_HF_PAPERS_PAYLOAD))
        hf._client = client

        papers = await hf.fetch_daily_papers()

        # Correct: 42 (from paper.upvotes), NOT 5 (from numComments)
        assert papers[0]["upvotes"] == "42"

    async def test_empty_list_returns_empty(self) -> None:
        import search.hf as hf

        client = _make_async_client(_mock_response(json_data=[]))
        hf._client = client

        papers = await hf.fetch_daily_papers()
        assert papers == []

    async def test_http_error_returns_empty_list(self) -> None:
        import httpx
        import search.hf as hf

        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.HTTPError("down"))
        hf._client = client

        papers = await hf.fetch_daily_papers()
        assert papers == []

    async def test_item_without_title_skipped(self) -> None:
        import search.hf as hf

        payload = [
            {"paper": {"id": "123", "title": "", "summary": "x", "upvotes": 1}, "numComments": 0}
        ]
        client = _make_async_client(_mock_response(json_data=payload))
        hf._client = client

        papers = await hf.fetch_daily_papers()
        assert papers == []

    def test_format_hf_papers_renders_upvotes(self) -> None:
        from search.hf import format_hf_papers

        papers = [
            {
                "title": "Big Paper",
                "summary": "Does great things.",
                "url": "https://hf.co/p/1",
                "upvotes": "99",
            },
        ]
        result = format_hf_papers(papers)
        assert "Big Paper" in result
        assert "99" in result
        assert "https://hf.co/p/1" in result

    def test_format_hf_papers_empty_returns_empty_string(self) -> None:
        from search.hf import format_hf_papers

        assert format_hf_papers([]) == ""
