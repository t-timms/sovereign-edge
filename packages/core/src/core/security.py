from __future__ import annotations

import re

# Common prompt injection / jailbreak patterns — defense in depth.
# Primary protection is always role-separated system prompts; this catches
# obvious attempts to override them via user-supplied text.
_INJECTION_RE = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions?|"
    r"you\s+are\s+now\s+|"
    r"new\s+(?:system\s+)?instructions?\s*:|"
    r"<\s*/?system\s*>|"
    r"\[/?SYSTEM\]|\[/?INST\]|"
    r"<\|im_start\|>|<\|im_end\|>|"
    r"disregard\s+(?:all\s+)?previous|"
    r"your\s+(?:true\s+)?(?:role|purpose)\s+is\s+|"
    r"OVERRIDE\s+(?:INSTRUCTIONS?|PROMPT|RULES?)|"
    r"---\s*END\s+OF\s+(?:SYSTEM|CONTEXT|INSTRUCTIONS?)",
    re.IGNORECASE,
)


def sanitize_input(text: str) -> str:
    """Redact prompt injection patterns from user-supplied text before LLM injection.

    Use this on any untrusted text that will be interpolated directly into a
    prompt string (user messages, scraped content, file uploads, etc.).
    """
    return _INJECTION_RE.sub("[removed]", text)
