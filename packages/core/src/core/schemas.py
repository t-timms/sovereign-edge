"""
Pydantic output schemas for structured LLM responses.

These schemas define the expected structure for expert outputs when structured
extraction is required. All experts can return free-form text (content field),
while specialized experts can populate additional typed fields.

Usage with instructor (optional dependency in sovereign-edge-llm):
    from instructor import from_litellm
    import litellm
    client = from_litellm(litellm.acompletion)
    result: CareerResponse = await client.chat.completions.create(
        model=..., messages=..., response_model=CareerResponse
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExpertResponse(BaseModel):
    """Base structured response — all experts can return this."""

    content: str = Field(description="Primary response text, formatted for Telegram Markdown.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)


class SpiritualResponse(ExpertResponse):
    """Spiritual expert: devotional with verse reference and prayer."""

    verse_ref: str = Field(
        default="",
        description="Scripture reference, e.g. 'Psalm 23:1 (KJV)'",
    )
    prayer_prompt: str = Field(
        default="",
        description="Closing prayer prompt (one sentence).",
    )


class CareerResponse(ExpertResponse):
    """Career expert: job listings and coaching advice."""

    job_count: int = Field(
        default=0,
        description="Number of distinct job matches presented.",
    )
    top_action: str = Field(
        default="",
        description="The single highest-value action to take today.",
    )


class IntelligenceResponse(ExpertResponse):
    """Intelligence expert: research synthesis."""

    paper_count: int = Field(
        default=0,
        description="Number of papers or articles referenced.",
    )
    key_insight: str = Field(
        default="",
        description="One-sentence takeaway from the research.",
    )


class CreativeResponse(ExpertResponse):
    """Creative expert: content output with format metadata."""

    format_used: str = Field(
        default="",
        description="Content format, e.g. 'LinkedIn post', 'thread', 'essay'.",
    )
    word_count: int = Field(
        default=0,
        description="Approximate word count of the generated content.",
    )
