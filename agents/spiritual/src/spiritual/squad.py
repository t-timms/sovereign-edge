"""
Spiritual squad — faith formation, prayer, scripture study, devotionals.

Grounds every response with live Bible verse retrieval via bible-api.com
(free, no auth) so scripture is always accurate and properly cited.
"""

from __future__ import annotations

import time

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

logger = get_logger(__name__, component="spiritual")

_SYSTEM_PROMPT = """\
You are the Spiritual Intelligence of Sovereign Edge — a contemplative guide
rooted in Christian faith.

You have access to live Bible verse lookups. When scripture is provided in the
context, quote it exactly as retrieved and cite book, chapter, and verse. Help
with scripture study, prayer composition, theological questions, and daily
devotionals. Respond with depth, warmth, and scriptural grounding.

Format scripture quotes in italics with full citation (e.g., *"For God so loved
the world..."* — John 3:16 KJV).\
"""

_DEVOTIONAL_PROMPT = """\
Using the scripture verse above as your anchor, write a brief morning devotional:
1. Quote the verse exactly.
2. 2-3 sentences of reflection connecting it to daily life.
3. A one-sentence prayer.
Keep it under 120 words. Warm and personal in tone.\
"""


class SpiritualSquad(BaseSquad):
    """Handles faith-formation tasks and generates morning devotionals."""

    @property
    def name(self) -> str:
        return SquadName.SPIRITUAL

    async def process(self, task: TaskRequest) -> TaskResult:
        from llm.gateway import get_gateway
        from search.bible import extract_reference, format_verse, lookup, random_verse

        gateway = get_gateway()
        t0 = time.monotonic()

        scripture_context = ""
        if task.routing == RoutingDecision.CLOUD:
            # Try to extract a specific reference from the user's message
            ref = extract_reference(task.content)
            verse = await lookup(ref) if ref else await random_verse()
            formatted = format_verse(verse)
            if formatted:
                scripture_context = f"Scripture:\n{formatted}"

        prior_turns: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                import json

                prior_turns = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *prior_turns,
            {
                "role": "user",
                "content": (
                    f"{scripture_context}\n\n---\n{task.content}"
                    if scripture_context
                    else task.content
                ),
            },
        ]

        result = await gateway.complete(
            messages=messages,
            max_tokens=1024,
            routing=task.routing,
            squad=self.name,
        )

        return TaskResult(
            task_id=task.task_id,
            squad=SquadName.SPIRITUAL,
            content=result["content"],
            model_used=result["model"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
        )

    async def morning_brief(self) -> str:
        from llm.gateway import get_gateway
        from search.bible import format_verse, random_verse

        gateway = get_gateway()

        verse = await random_verse()
        verse_text = format_verse(verse)

        result = await gateway.complete(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Scripture:\n{verse_text}\n\n---\n{_DEVOTIONAL_PROMPT}"
                        if verse_text
                        else (
                            "Generate a brief morning devotional with a scripture verse, "
                            "2-3 sentences of reflection, and a one-sentence prayer. "
                            "Under 120 words."
                        )
                    ),
                },
            ],
            max_tokens=300,
            routing=RoutingDecision.CLOUD,
            squad=self.name,
        )
        return result["content"]

    async def health_check(self) -> bool:
        try:
            from llm.gateway import get_gateway

            result = await get_gateway().complete(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(result.get("content"))
        except Exception:
            logger.warning("spiritual_health_check_failed", exc_info=True)
            return False
