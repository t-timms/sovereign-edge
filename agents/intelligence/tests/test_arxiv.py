from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from intelligence.arxiv import (
    Paper,
    format_digest,
    get_hf_daily_papers,
    get_research_digest,
    search_arxiv,
)

# ── Paper dataclass ───────────────────────────────────────────────────────────


def test_paper_fields() -> None:
    p = Paper(
        title="Attention Is All You Need",
        authors="Vaswani et al.",
        abstract="We propose Transformer...",
        url="https://arxiv.org/abs/1706.03762",
        published="2017-06-12",
        source="arxiv",
    )
    assert p.title == "Attention Is All You Need"
    assert p.source == "arxiv"


# ── search_arxiv ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_search_arxiv_returns_papers() -> None:
    atom_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>LLM Reasoning Survey</title>
    <summary>A comprehensive survey of reasoning in large language models.</summary>
    <link href="https://arxiv.org/abs/2401.00001"/>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
  </entry>
</feed>"""

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = atom_feed

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("intelligence.arxiv.httpx.AsyncClient", return_value=mock_client):
        papers = await search_arxiv("LLM reasoning")

    assert len(papers) == 1
    assert papers[0].title == "LLM Reasoning Survey"
    assert papers[0].source == "arxiv"
    assert "Alice Smith" in papers[0].authors


@pytest.mark.asyncio()
async def test_search_arxiv_returns_empty_on_http_error() -> None:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=RuntimeError("connection refused"))

    with patch("intelligence.arxiv.httpx.AsyncClient", return_value=mock_client):
        papers = await search_arxiv("query")

    assert papers == []


@pytest.mark.asyncio()
async def test_search_arxiv_strips_newlines_from_title() -> None:
    atom_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Multi-line\nTitle Here</title>
    <summary>Abstract text.</summary>
    <link href="https://arxiv.org/abs/2401.00002"/>
    <published>2024-01-02T00:00:00Z</published>
  </entry>
</feed>"""

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = atom_feed

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("intelligence.arxiv.httpx.AsyncClient", return_value=mock_client):
        papers = await search_arxiv("multi-line")

    assert "\n" not in papers[0].title


# ── get_hf_daily_papers ───────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_get_hf_daily_papers_parses_response() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = [
        {
            "paper": {
                "title": "HF Daily Paper",
                "authors": [{"name": "Bob Jones"}],
                "summary": "A daily paper from HF.",
                "id": "2401.99999",
                "publishedAt": "2026-04-03",
            }
        }
    ]

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("intelligence.arxiv.httpx.AsyncClient", return_value=mock_client):
        papers = await get_hf_daily_papers(limit=5)

    assert len(papers) == 1
    assert papers[0].title == "HF Daily Paper"
    assert papers[0].source == "huggingface"
    assert "2401.99999" in papers[0].url


@pytest.mark.asyncio()
async def test_get_hf_daily_papers_returns_empty_on_error() -> None:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=RuntimeError("network error"))

    with patch("intelligence.arxiv.httpx.AsyncClient", return_value=mock_client):
        papers = await get_hf_daily_papers()

    assert papers == []


@pytest.mark.asyncio()
async def test_get_hf_daily_papers_respects_limit() -> None:
    items = [
        {
            "paper": {
                "title": f"Paper {i}",
                "authors": [],
                "summary": "",
                "id": f"000{i}",
                "publishedAt": "",
            }
        }
        for i in range(10)
    ]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = items

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("intelligence.arxiv.httpx.AsyncClient", return_value=mock_client):
        papers = await get_hf_daily_papers(limit=3)

    assert len(papers) == 3


# ── get_research_digest ───────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_get_research_digest_deduplicates_titles() -> None:
    dup_paper = Paper(
        title="Duplicate Paper Title",
        authors="A",
        abstract="x",
        url="http://a",
        published="2026",
        source="arxiv",
    )
    with (
        patch(
            "intelligence.arxiv.search_arxiv", new=AsyncMock(return_value=[dup_paper, dup_paper])
        ),
        patch("intelligence.arxiv.get_hf_daily_papers", new=AsyncMock(return_value=[])),
    ):
        results = await get_research_digest(queries=["test"])

    # Duplicate title should be merged into one
    titles = [p.title for p in results]
    assert titles.count("Duplicate Paper Title") == 1


@pytest.mark.asyncio()
async def test_get_research_digest_caps_at_15() -> None:
    papers = [
        Paper(title=f"Paper {i}", authors="", abstract="", url="", published="", source="arxiv")
        for i in range(20)
    ]
    with (
        patch("intelligence.arxiv.search_arxiv", new=AsyncMock(return_value=papers)),
        patch("intelligence.arxiv.get_hf_daily_papers", new=AsyncMock(return_value=[])),
    ):
        results = await get_research_digest(queries=["single"])

    assert len(results) <= 15


# ── format_digest ─────────────────────────────────────────────────────────────


def test_format_digest_empty() -> None:
    assert format_digest([]) == "No recent papers found."


def test_format_digest_includes_title_and_url() -> None:
    papers = [
        Paper(
            title="Test Paper",
            authors="John Doe",
            abstract="This paper tests things.",
            url="https://arxiv.org/abs/0000",
            published="2026-04-03",
            source="arxiv",
        )
    ]
    result = format_digest(papers)
    assert "Test Paper" in result
    assert "https://arxiv.org/abs/0000" in result
    assert "John Doe" in result
