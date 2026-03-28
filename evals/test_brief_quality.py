"""
Brief quality evals — assert structural integrity of expert outputs.

These are not unit tests of internal logic; they test output contracts:
the things a degraded or hallucinating LLM would violate. Run alongside
the main test suite via `pytest evals/`.
"""

from __future__ import annotations

import re

import pytest

# ---------------------------------------------------------------------------
# Helpers (mirrors BriefOutput logic without importing the expert)
# ---------------------------------------------------------------------------

def _has_link(text: str) -> bool:
    return bool(re.search(r"https?://\S+", text))


def _word_count(text: str) -> int:
    return len(text.split())


def _has_arxiv_link(text: str) -> bool:
    return bool(re.search(r"https?://arxiv\.org/\S+", text))


def _has_hf_link(text: str) -> bool:
    return bool(re.search(r"https?://huggingface\.co/\S+", text))


_DFW_COMPANIES = {
    "capital one", "at&t", "american airlines", "sabre", "jpmorgan", "chase",
    "dell", "toyota", "deloitte", "accenture", "lockheed", "raytheon",
    "amazon", "google", "microsoft", "tesla",
}


def _has_company_name(text: str) -> bool:
    lower = text.lower()
    return any(c in lower for c in _DFW_COMPANIES)


# ---------------------------------------------------------------------------
# Intelligence brief contracts
# ---------------------------------------------------------------------------

class TestIntelligenceBriefContracts:
    """A valid intelligence brief must have at least one link and be substantive."""

    GOOD_BRIEF = (
        "*Transformer Attention Survey* — new benchmark results on long-context tasks "
        "show MLA attention cuts KV cache by 93%.\n\n"
        "[arXiv:2501.12345](https://arxiv.org/abs/2501.12345)\n\n"
        "*HuggingFace Daily Paper*: multimodal RAG with ColQwen2 now outperforms "
        "OCR-based pipelines on document retrieval. "
        "[paper](https://huggingface.co/papers/2501.99999)\n\n"
        "*Action item*: test MLA on the Jetson — should cut VRAM pressure on 16GB."
    )

    EMPTY_BRIEF = ""
    LINKLESS_BRIEF = "Some AI things happened today. Very interesting. No sources cited."
    WALL_OF_TEXT = " ".join(["word"] * 400)

    def test_good_brief_passes_all_checks(self) -> None:
        assert _has_link(self.GOOD_BRIEF)
        assert _has_arxiv_link(self.GOOD_BRIEF)
        assert _word_count(self.GOOD_BRIEF) < 300
        assert self.GOOD_BRIEF.strip()

    def test_empty_brief_fails_link_check(self) -> None:
        assert not _has_link(self.EMPTY_BRIEF)
        assert _word_count(self.EMPTY_BRIEF) == 0

    def test_linkless_brief_flagged(self) -> None:
        assert not _has_link(self.LINKLESS_BRIEF)

    def test_wall_of_text_exceeds_word_limit(self) -> None:
        assert _word_count(self.WALL_OF_TEXT) >= 300

    def test_brief_output_model_validates(self) -> None:
        from intelligence.expert import BriefOutput

        good = BriefOutput(content=self.GOOD_BRIEF)
        assert good.is_valid
        assert good.link_count >= 2
        assert good.word_count < 300

        bad = BriefOutput(content=self.LINKLESS_BRIEF)
        assert not bad.is_valid
        assert bad.link_count == 0

    def test_brief_output_empty_content_invalid(self) -> None:
        from intelligence.expert import BriefOutput

        empty = BriefOutput(content="")
        assert not empty.is_valid


# ---------------------------------------------------------------------------
# Career brief contracts
# ---------------------------------------------------------------------------

class TestCareerBriefContracts:
    """Career briefs should mention at least one company and have a job title."""

    GOOD_CAREER = (
        "*ML Engineer — Capital One* (Plano, TX)\n\n"
        "Remote-friendly. 5 YOE preferred. PyTorch, LLM serving, AWS. "
        "Salary range $160K-$200K. Apply via LinkedIn.\n\n"
        "*AI Solutions Engineer — AT&T* (Dallas, TX)\n\n"
        "LangGraph, FastAPI, RAG pipelines. Hybrid. $140K-$175K."
    )

    NO_COMPANY_BRIEF = "There are some job opportunities available in the tech sector."

    def test_good_career_brief_has_company(self) -> None:
        assert _has_company_name(self.GOOD_CAREER)

    def test_no_company_career_brief_flagged(self) -> None:
        assert not _has_company_name(self.NO_COMPANY_BRIEF)

    def test_career_brief_mentions_salary_or_role(self) -> None:
        keywords = {"engineer", "scientist", "analyst", "developer", "architect"}
        assert any(kw in self.GOOD_CAREER.lower() for kw in keywords)


# ---------------------------------------------------------------------------
# Markdown sanitizer contracts (bot.py _sanitize_markdown)
# ---------------------------------------------------------------------------

class TestMarkdownSanitizer:
    """Telegram MarkdownV1 only renders *bold* and _italic_ — verify sanitizer."""

    def test_double_asterisk_converted(self) -> None:
        from telegram_bot.bot import _sanitize_markdown

        result = _sanitize_markdown("**bold text** in a sentence")
        assert "**" not in result
        assert "<b>bold text</b>" in result

    def test_hash_headers_stripped(self) -> None:
        from telegram_bot.bot import _sanitize_markdown

        result = _sanitize_markdown("## My Header\nsome content")
        assert "##" not in result
        assert "My Header" in result

    def test_dividers_stripped(self) -> None:
        from telegram_bot.bot import _sanitize_markdown

        result = _sanitize_markdown("above\n---\nbelow")
        assert "---" not in result

    def test_bare_url_wrapped(self) -> None:
        from telegram_bot.bot import _sanitize_markdown

        result = _sanitize_markdown("See https://arxiv.org/abs/2501.12345 for details")
        assert '<a href="https://arxiv.org/abs/2501.12345">' in result

    def test_existing_markdown_link_not_double_wrapped(self) -> None:
        from telegram_bot.bot import _sanitize_markdown

        original = "See [paper](https://arxiv.org/abs/2501.12345) here"
        result = _sanitize_markdown(original)
        # Should not produce [[paper](url)](url)
        assert "[[" not in result


# ---------------------------------------------------------------------------
# Router classifier contracts
# ---------------------------------------------------------------------------

class TestRouterClassifier:
    """Keyword classifier should route clear signals correctly."""

    @pytest.fixture()
    def router(self):
        from router.classifier import IntentRouter

        return IntentRouter()  # bootstrap (keyword) mode — no ONNX needed

    def test_bible_routes_to_spiritual(self, router) -> None:
        from core.types import Intent

        intent, confidence = router.classify("What does the Bible say about forgiveness?")
        assert intent == Intent.SPIRITUAL
        assert confidence > 0.6

    def test_job_search_routes_to_career(self, router) -> None:
        from core.types import Intent

        intent, confidence = router.classify("Find me ML engineer jobs in Dallas")
        assert intent == Intent.CAREER
        assert confidence > 0.6

    def test_arxiv_routes_to_intelligence(self, router) -> None:
        from core.types import Intent

        intent, confidence = router.classify("Summarize recent arXiv papers on transformers")
        assert intent == Intent.INTELLIGENCE
        assert confidence > 0.6

    def test_linkedin_post_routes_to_creative(self, router) -> None:
        from core.types import Intent

        intent, confidence = router.classify("Write a LinkedIn post about my AI project")
        assert intent == Intent.CREATIVE
        assert confidence > 0.6

    def test_ambiguous_query_returns_general_low_confidence(self, router) -> None:
        from core.types import Intent
        from router.classifier import LOW_CONFIDENCE_THRESHOLD

        intent, confidence = router.classify("hey what do you think")
        assert intent == Intent.GENERAL
        assert confidence <= LOW_CONFIDENCE_THRESHOLD
