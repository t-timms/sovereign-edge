"""Regex-based PII detection. Zero ML overhead, ~0MB RAM."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIMatch:
    """A detected PII instance."""

    type: str
    value: str
    start: int
    end: int


# Patterns ordered by specificity (most specific first)
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
    ),
]


class PIIDetector:
    """Detects personally identifiable information in text."""

    def contains_pii(self, text: str) -> bool:
        """Fast check — returns True if any PII pattern matches."""
        return any(pattern.search(text) for _, pattern in _PII_PATTERNS)

    def detect_all(self, text: str) -> list[PIIMatch]:
        """Full scan — returns all PII matches with types and positions."""
        matches: list[PIIMatch] = []
        for pii_type, pattern in _PII_PATTERNS:
            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return matches

    def redact(self, text: str) -> str:
        """Replace PII with type labels for safe cloud transmission."""
        result = text
        for pii_type, pattern in _PII_PATTERNS:
            result = pattern.sub(f"[{pii_type}_REDACTED]", result)
        return result
