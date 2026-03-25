"""
Multi-provider LLM gateway using LiteLLM as a library (NOT proxy).

Automatic fallback chain: Groq → Gemini → Cerebras → Mistral → Local
Cost tracking and rate limit enforcement per provider per day.
~75MB RAM overhead.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

import litellm
from core.config import get_settings
from core.types import RoutingDecision

# Suppress LiteLLM's verbose logging
litellm.set_verbose = False  # type: ignore[attr-defined]
litellm.suppress_debug_info = True  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    model: str
    rpm: int        # Requests per minute
    tpd: int        # Tokens per day (approximate limit)
    priority: int   # Lower = tried first
    env_key: str    # Environment variable name for API key


# Provider chain ordered by priority
PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        model="groq/llama-3.3-70b-versatile",
        rpm=30, tpd=500_000, priority=1, env_key="GROQ_API_KEY",
    ),
    ProviderConfig(
        model="gemini/gemini-2.5-flash-preview-04-17",
        rpm=15, tpd=250_000, priority=2, env_key="GOOGLE_API_KEY",
    ),
    ProviderConfig(
        model="cerebras/llama-3.3-70b",
        rpm=30, tpd=1_000_000, priority=3, env_key="CEREBRAS_API_KEY",
    ),
    ProviderConfig(
        model="mistral/mistral-small-latest",
        rpm=2, tpd=33_000_000, priority=4, env_key="MISTRAL_API_KEY",
    ),
]

LOCAL_FALLBACK_MODEL = "ollama/qwen3:0.6b"


@dataclass
class UsageTracker:
    """Tracks daily token usage per provider."""

    date: date = field(default_factory=date.today)
    usage: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, tokens: int) -> None:
        self._reset_if_new_day()
        self.usage[model] = self.usage.get(model, 0) + tokens

    def get(self, model: str) -> int:
        self._reset_if_new_day()
        return self.usage.get(model, 0)

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if self.date != today:
            self.usage.clear()
            self.date = today


class LLMGateway:
    """
    Multi-provider LLM gateway.

    Usage:
        gateway = LLMGateway()
        response = await gateway.complete(
            messages=[{"role": "user", "content": "Hello"}],
            routing=RoutingDecision.CLOUD,
        )
    """

    def __init__(self) -> None:
        self.tracker = UsageTracker()
        self.settings = get_settings()

    async def complete(
        self,
        messages: list[dict[str, str]],
        routing: RoutingDecision = RoutingDecision.CLOUD,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        squad: str = "general",
    ) -> dict[str, object]:
        """
        Send a completion request through the fallback chain.

        Returns dict with keys: content, model, tokens_in, tokens_out, latency_ms, cost_usd
        """
        if routing == RoutingDecision.LOCAL:
            return await self._call_local(messages, max_tokens, temperature, squad)

        # Try cloud providers in priority order
        for provider in sorted(PROVIDERS, key=lambda p: p.priority):
            if self.tracker.get(provider.model) >= provider.tpd:
                logger.debug("Provider %s daily limit reached, skipping", provider.model)
                continue

            try:
                start = time.monotonic()
                response = await litellm.acompletion(  # type: ignore[attr-defined]
                    model=provider.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=30,
                )
                elapsed = (time.monotonic() - start) * 1000

                tokens_in = response.usage.prompt_tokens if response.usage else 0
                tokens_out = response.usage.completion_tokens if response.usage else 0
                total_tokens = tokens_in + tokens_out
                self.tracker.add(provider.model, total_tokens)

                cost = self._estimate_cost(provider.model, tokens_in, tokens_out)
                content = response.choices[0].message.content or ""

                logger.info(
                    "LLM response",
                    extra={
                        "model": provider.model,
                        "squad": squad,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "latency_ms": round(elapsed, 1),
                        "cost_usd": cost,
                    },
                )

                return {
                    "content": content,
                    "model": provider.model,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "latency_ms": round(elapsed, 1),
                    "cost_usd": cost,
                }

            except Exception as e:
                logger.warning("Provider %s failed: %s", provider.model, e)
                continue

        # All cloud providers failed — fall back to local
        logger.warning("All cloud providers failed, falling back to local model")
        return await self._call_local(messages, max_tokens, temperature, squad)

    async def _call_local(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        squad: str,
    ) -> dict[str, object]:
        """Call local Ollama model as last resort."""
        start = time.monotonic()
        try:
            response = await litellm.acompletion(  # type: ignore[attr-defined]
                model=LOCAL_FALLBACK_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=120,  # Local models are slower
                api_base=self.settings.ollama_host,
            )
            elapsed = (time.monotonic() - start) * 1000
            content = response.choices[0].message.content or ""
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0

            return {
                "content": content,
                "model": LOCAL_FALLBACK_MODEL,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": round(elapsed, 1),
                "cost_usd": 0.0,  # Local is free
            }
        except Exception as e:
            logger.error("Local model also failed: %s", e, exc_info=True)
            return {
                "content": f"[ERROR] All inference providers failed. Last error: {e}",
                "model": "none",
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": (time.monotonic() - start) * 1000,
                "cost_usd": 0.0,
            }

    @staticmethod
    def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
        """Rough cost estimate. All free-tier models return 0."""
        # All providers in our chain are free tier
        return 0.0
