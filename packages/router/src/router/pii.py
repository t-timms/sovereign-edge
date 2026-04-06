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


def _luhn_valid(value: str) -> bool:
    """Return True if the digit sequence passes the Luhn checksum (credit card validation).

    Prevents false positives on 16-digit sequences such as ISBNs, tracking numbers,
    and model IDs — real credit card numbers must satisfy the Luhn algorithm.
    """
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


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
        for pii_type, pattern in _PII_PATTERNS:
            for match in pattern.finditer(text):
                if pii_type == "CREDIT_CARD" and not _luhn_valid(match.group()):
                    continue
                return True
        return False

    def detect_all(self, text: str) -> list[PIIMatch]:
        """Full scan — returns all PII matches with types and positions."""
        matches: list[PIIMatch] = []
        for pii_type, pattern in _PII_PATTERNS:
            for match in pattern.finditer(text):
                if pii_type == "CREDIT_CARD" and not _luhn_valid(match.group()):
                    continue
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
            if pii_type == "CREDIT_CARD":
                result = pattern.sub(
                    lambda m, _t=pii_type: (
                        f"[{_t}_REDACTED]" if _luhn_valid(m.group()) else m.group()
                    ),
                    result,
                )
            else:
                result = pattern.sub(f"[{pii_type}_REDACTED]", result)
        return result
