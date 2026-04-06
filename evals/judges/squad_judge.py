"""Generic squad output judge — heuristic checks on LLM response quality.

Checks:
  - min_length: response has minimum character count
  - must_contain: case-insensitive substrings that must appear
  - must_not_contain: substrings that must NOT appear (hallucination guards)
  - must_cite_verse: if True, response must contain a book:chapter:verse pattern
  - must_include_number: if True, response must contain at least one number
  - is_devotional: if True, response must contain common devotional keywords
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_VERSE_RE = re.compile(r"\d+:\d+")  # e.g. "3:16", "23:1"
_NUMBER_RE = re.compile(r"\d+")
_DEVOTIONAL_KEYWORDS = {"verse", "reflect", "apply", "prayer", "today", "lord", "amen", "god"}


@dataclass
class SquadEvalResult:
    query: str
    response: str
    passed: bool
    failures: list[str] = field(default_factory=list)


def check(query: str, response: str, criteria: dict) -> SquadEvalResult:
    """Evaluate one squad response against its criteria dict."""
    failures: list[str] = []
    resp_lower = response.lower()

    min_len = criteria.get("min_length", 0)
    if len(response) < min_len:
        failures.append(f"Too short: {len(response)} < {min_len} chars")

    for kw in criteria.get("must_contain", []):
        if kw.lower() not in resp_lower:
            failures.append(f"Missing required term: '{kw}'")

    for kw in criteria.get("must_not_contain", []):
        if kw.lower() in resp_lower:
            failures.append(f"Contains forbidden term: '{kw}'")

    if criteria.get("must_cite_verse") and not _VERSE_RE.search(response):
        failures.append("No verse citation found (expected X:Y pattern)")

    if criteria.get("must_include_number") and not _NUMBER_RE.search(response):
        failures.append("No number found in response")

    if criteria.get("is_devotional"):
        found = sum(1 for kw in _DEVOTIONAL_KEYWORDS if kw in resp_lower)
        if found < 2:
            failures.append(f"Doesn't read like a devotional ({found}/2 keywords)")

    return SquadEvalResult(
        query=query,
        response=response[:200],
        passed=len(failures) == 0,
        failures=failures,
    )


def score(results: list[SquadEvalResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total > 0 else 0.0,
        "failures": [
            {"query": r.query[:60], "failures": r.failures, "response_preview": r.response[:100]}
            for r in results
            if not r.passed
        ],
    }


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
