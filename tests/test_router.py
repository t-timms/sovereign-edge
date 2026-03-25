"""Tests for PII detection and keyword intent classifier."""

from __future__ import annotations

import pytest
from core.types import Intent, RoutingDecision
from router.classifier import LOW_CONFIDENCE_THRESHOLD, IntentRouter
from router.pii import PIIDetector

# ─────────────────────────────────────────────────────────────────────────────
# PIIDetector
# ─────────────────────────────────────────────────────────────────────────────


class TestPIIDetector:
    def setup_method(self) -> None:
        self.detector = PIIDetector()

    # ── contains_pii ──────────────────────────────────────────────────────────

    def test_detects_email(self) -> None:
        assert self.detector.contains_pii("Contact me at john.doe@example.com please") is True

    def test_detects_ssn(self) -> None:
        assert self.detector.contains_pii("My SSN is 123-45-6789.") is True

    def test_detects_us_phone(self) -> None:
        assert self.detector.contains_pii("Call me at 555-867-5309.") is True

    def test_detects_credit_card(self) -> None:
        assert self.detector.contains_pii("Card: 4111 1111 1111 1111") is True

    def test_detects_ip_address(self) -> None:
        assert self.detector.contains_pii("Server is at 192.168.1.1") is True

    def test_clean_text_has_no_pii(self) -> None:
        assert self.detector.contains_pii("What is the best PyTorch optimizer?") is False

    def test_empty_string_has_no_pii(self) -> None:
        assert self.detector.contains_pii("") is False

    # ── False-positive guard ──────────────────────────────────────────────────

    def test_version_number_not_flagged_as_ssn(self) -> None:
        """'3.11.2' has hyphens but not SSN pattern — must not false-positive."""
        assert self.detector.contains_pii("Using Python 3.11.2 with PyTorch") is False

    def test_model_name_not_email(self) -> None:
        assert self.detector.contains_pii("llama-3.3-70b is a great model") is False

    # ── detect_all ────────────────────────────────────────────────────────────

    def test_detect_all_returns_correct_types(self) -> None:
        matches = self.detector.detect_all("Email alice@test.com and SSN 987-65-4321")
        types = {m.type for m in matches}
        assert "EMAIL" in types
        assert "SSN" in types

    def test_detect_all_empty_text_returns_empty_list(self) -> None:
        assert self.detector.detect_all("") == []

    def test_detect_all_match_positions_are_valid(self) -> None:
        text = "Reach me at bob@mail.org today"
        matches = self.detector.detect_all(text)
        for m in matches:
            assert m.start >= 0
            assert m.end <= len(text)
            assert text[m.start : m.end] == m.value

    # ── redact ────────────────────────────────────────────────────────────────

    def test_redact_replaces_email(self) -> None:
        result = self.detector.redact("Email me at foo@bar.com tomorrow")
        assert "foo@bar.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_redact_preserves_non_pii_text(self) -> None:
        result = self.detector.redact("Hello world, no PII here")
        assert result == "Hello world, no PII here"

    def test_redact_handles_multiple_pii_types(self) -> None:
        text = "SSN 111-22-3333 and email test@x.com"
        result = self.detector.redact(text)
        assert "111-22-3333" not in result
        assert "test@x.com" not in result


# ─────────────────────────────────────────────────────────────────────────────
# IntentRouter — keyword mode (no ONNX model loaded)
# ─────────────────────────────────────────────────────────────────────────────


class TestIntentRouterKeywords:
    def setup_method(self) -> None:
        # No model_path → forces keyword mode
        self.router = IntentRouter()

    def test_spiritual_keyword(self) -> None:
        intent, _ = self.router.classify("What does the Bible say about forgiveness?")
        assert intent == Intent.SPIRITUAL

    def test_career_keyword(self) -> None:
        intent, _ = self.router.classify("Help me update my resume for a data science job")
        assert intent == Intent.CAREER

    def test_intelligence_keyword(self) -> None:
        intent, _ = self.router.classify("Summarize the latest arxiv papers on transformers")
        assert intent == Intent.INTELLIGENCE

    def test_creative_keyword(self) -> None:
        intent, _ = self.router.classify("Write a YouTube script about GRPO training")
        assert intent == Intent.CREATIVE

    def test_general_returns_default_confidence(self) -> None:
        intent, confidence = self.router.classify("How are you today?")
        assert intent == Intent.GENERAL
        assert confidence == pytest.approx(0.5)

    def test_confidence_above_zero_for_matched_intent(self) -> None:
        _, confidence = self.router.classify("Pray for wisdom")
        assert confidence > 0.5

    def test_confidence_capped_at_0_95(self) -> None:
        # Very keyword-heavy text — should still cap at 0.95
        text = "bible verse prayer scripture devotion faith church worship sermon"
        _, confidence = self.router.classify(text)
        assert confidence <= 0.95

    # ── Multi-word priority ───────────────────────────────────────────────────

    def test_cover_letter_maps_to_career_not_creative(self) -> None:
        """'cover letter' is a 2-word key → CAREER; 'write' alone → CREATIVE.
        Multi-word key must win when both are present."""
        intent, _ = self.router.classify("Write me a cover letter for this job")
        assert intent == Intent.CAREER

    def test_machine_learning_maps_to_intelligence(self) -> None:
        intent, _ = self.router.classify("Explain machine learning to me")
        assert intent == Intent.INTELLIGENCE

    def test_linkedin_post_maps_to_creative(self) -> None:
        intent, _ = self.router.classify("Draft a LinkedIn post about my project")
        assert intent == Intent.CREATIVE


# ─────────────────────────────────────────────────────────────────────────────
# IntentRouter.route — PII forces LOCAL; clean text stays CLOUD
# ─────────────────────────────────────────────────────────────────────────────


class TestIntentRouterRouting:
    def setup_method(self) -> None:
        self.router = IntentRouter()

    def test_pii_text_routes_to_local(self) -> None:
        _, _, routing = self.router.route("My email is secret@private.com check my resume")
        assert routing == RoutingDecision.LOCAL

    def test_clean_text_routes_to_cloud(self) -> None:
        _, _, routing = self.router.route("What is the best Python framework for REST APIs?")
        assert routing == RoutingDecision.CLOUD

    def test_route_returns_correct_intent_alongside_local(self) -> None:
        """PII in a career question → LOCAL but intent is still CAREER."""
        intent, _, routing = self.router.route("Review my resume, SSN 999-88-7777")
        assert intent == Intent.CAREER
        assert routing == RoutingDecision.LOCAL

    def test_low_confidence_general_still_routes_cloud(self) -> None:
        """Low confidence does not redirect to LOCAL — it stays CLOUD."""
        intent, confidence, routing = self.router.route("meh")
        assert intent == Intent.GENERAL
        assert confidence <= LOW_CONFIDENCE_THRESHOLD
        assert routing == RoutingDecision.CLOUD
