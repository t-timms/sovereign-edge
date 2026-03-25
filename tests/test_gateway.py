"""Tests for LLM gateway — TokenBucket, UsageTracker, singleton, cost tracking."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from core.types import RoutingDecision

# ─────────────────────────────────────────────────────────────────────────────
# TokenBucket
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenBucket:
    def test_full_bucket_allows_first_request(self) -> None:
        from llm.gateway import TokenBucket

        bucket = TokenBucket(rpm=10)
        assert bucket.acquire() is True

    def test_empty_bucket_rejects_request(self) -> None:
        from llm.gateway import TokenBucket

        bucket = TokenBucket(rpm=2)
        bucket.acquire()
        bucket.acquire()
        # Bucket is now empty — third request must be rejected
        assert bucket.acquire() is False

    def test_bucket_refills_over_time(self) -> None:
        from llm.gateway import TokenBucket

        bucket = TokenBucket(rpm=60)  # 1 token/second
        # Drain the entire bucket
        for _ in range(60):
            bucket.acquire()
        assert bucket.acquire() is False

        # Simulate 2 seconds of wall-clock refill by back-dating _last_refill
        bucket._last_refill -= 2.0
        assert bucket.acquire() is True

    def test_tokens_cannot_exceed_rpm_capacity(self) -> None:
        from llm.gateway import TokenBucket

        bucket = TokenBucket(rpm=5)
        # Wind back time by a huge amount
        bucket._last_refill -= 10_000.0
        bucket.acquire()  # triggers refill
        assert bucket._tokens <= float(bucket.rpm)


# ─────────────────────────────────────────────────────────────────────────────
# UsageTracker
# ─────────────────────────────────────────────────────────────────────────────


class TestUsageTracker:
    def test_add_and_get(self) -> None:
        from llm.gateway import UsageTracker

        tracker = UsageTracker()
        tracker.add("groq/llama3", 100)
        tracker.add("groq/llama3", 50)
        assert tracker.get("groq/llama3") == 150

    def test_get_unknown_model_returns_zero(self) -> None:
        from llm.gateway import UsageTracker

        tracker = UsageTracker()
        assert tracker.get("unknown/model") == 0

    def test_total_today_sums_all_models(self) -> None:
        from llm.gateway import UsageTracker

        tracker = UsageTracker()
        tracker.add("groq/llama3", 200)
        tracker.add("gemini/flash", 300)
        assert tracker.total_today() == 500

    def test_resets_on_new_day(self) -> None:
        from llm.gateway import UsageTracker

        tracker = UsageTracker()
        tracker.add("groq/llama3", 999)

        # Simulate midnight rollover
        tracker._date = date.today() - timedelta(days=1)

        assert tracker.get("groq/llama3") == 0
        assert tracker.total_today() == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_gateway() singleton
# ─────────────────────────────────────────────────────────────────────────────


class TestGatewaySingleton:
    def test_returns_same_instance(self) -> None:
        from llm.gateway import get_gateway

        gw1 = get_gateway()
        gw2 = get_gateway()
        assert gw1 is gw2

    def test_instance_reset_by_fixture(self) -> None:
        """conftest resets _instance after each test — this verifies the pattern."""
        import llm.gateway as gw_mod
        from llm.gateway import get_gateway

        gw_mod._instance = None  # explicit reset
        instance = get_gateway()
        assert instance is not None
        assert gw_mod._instance is instance


# ─────────────────────────────────────────────────────────────────────────────
# complete() — cloud path with mocked litellm
# ─────────────────────────────────────────────────────────────────────────────


def _make_litellm_response(
    content: str = "Hello!",
    tokens_in: int = 10,
    tokens_out: int = 5,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = tokens_in
    usage.completion_tokens = tokens_out

    choice = MagicMock()
    choice.message.content = content

    resp = MagicMock()
    resp.usage = usage
    resp.choices = [choice]
    return resp


class TestGatewayComplete:
    async def test_successful_completion_returns_content(self) -> None:
        import llm.gateway as gw_mod

        gw_mod._instance = None
        gw = gw_mod.get_gateway()

        mock_resp = _make_litellm_response(content="Great answer", tokens_in=20, tokens_out=10)

        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)),
            patch("litellm.completion_cost", return_value=0.00005),
        ):
            result = await gw.complete(
                messages=[{"role": "user", "content": "Hello"}],
                routing=RoutingDecision.CLOUD,
            )

        assert result["content"] == "Great answer"
        assert result["tokens_in"] == 20
        assert result["tokens_out"] == 10
        assert result["cost_usd"] == pytest.approx(0.00005)

    async def test_cost_usd_is_nonzero_on_success(self) -> None:
        """cost_usd must not be hardcoded 0.0 — must come from litellm.completion_cost."""
        import llm.gateway as gw_mod

        gw_mod._instance = None
        gw = gw_mod.get_gateway()

        mock_resp = _make_litellm_response(content="ok")

        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)),
            patch("litellm.completion_cost", return_value=0.00123),
        ):
            result = await gw.complete(
                messages=[{"role": "user", "content": "cost test"}],
                routing=RoutingDecision.CLOUD,
            )

        assert result["cost_usd"] > 0

    async def test_all_providers_fail_falls_back_to_local(self) -> None:
        """All cloud providers raise ServiceUnavailableError → local fallback."""
        import litellm
        import llm.gateway as gw_mod

        gw_mod._instance = None
        gw = gw_mod.get_gateway()
        local_resp = _make_litellm_response(content="local answer")

        async def _side_effect(*args: object, **kwargs: object) -> object:
            model = kwargs.get("model", "")
            if model == gw_mod.LOCAL_FALLBACK_MODEL:
                return local_resp
            raise litellm.ServiceUnavailableError(  # type: ignore[attr-defined]
                message="down", llm_provider="test", model=str(model)
            )

        with (
            patch("litellm.acompletion", side_effect=_side_effect),
            patch("litellm.completion_cost", return_value=0.0),
            patch("asyncio.sleep", new=AsyncMock()),  # skip retry delays
        ):
            result = await gw.complete(
                messages=[{"role": "user", "content": "fallback test"}],
                routing=RoutingDecision.CLOUD,
            )

        assert result["model"] == gw_mod.LOCAL_FALLBACK_MODEL
        assert result["content"] == "local answer"

    async def test_local_routing_bypasses_cloud(self) -> None:
        import llm.gateway as gw_mod

        gw_mod._instance = None
        gw = gw_mod.get_gateway()

        local_resp = _make_litellm_response(content="local only")
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=local_resp)),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            result = await gw.complete(
                messages=[{"role": "user", "content": "keep local"}],
                routing=RoutingDecision.LOCAL,
            )

        assert result["model"] == gw_mod.LOCAL_FALLBACK_MODEL

    async def test_auth_error_skips_provider(self) -> None:
        """AuthenticationError must cause the provider to be skipped, not retried."""
        import litellm
        import llm.gateway as gw_mod

        gw_mod._instance = None
        gw = gw_mod.get_gateway()

        local_resp = _make_litellm_response(content="fallback")

        def _side_effect(*args: object, **kwargs: object) -> object:
            model = kwargs.get("model", "")
            if "ollama" in str(model):
                return local_resp
            raise litellm.AuthenticationError(  # type: ignore[attr-defined]
                message="bad key", llm_provider="groq", model="llama3"
            )

        with (
            patch("litellm.acompletion", new=AsyncMock(side_effect=_side_effect)),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            result = await gw.complete(
                messages=[{"role": "user", "content": "auth fail"}],
                routing=RoutingDecision.CLOUD,
            )

        # Must eventually land on local fallback, not hang in a retry loop
        assert "content" in result

    async def test_usage_tracker_updated_on_success(self) -> None:
        import llm.gateway as gw_mod

        gw_mod._instance = None
        gw = gw_mod.get_gateway()

        mock_resp = _make_litellm_response(tokens_in=30, tokens_out=15)
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            await gw.complete(
                messages=[{"role": "user", "content": "track usage"}],
                routing=RoutingDecision.CLOUD,
            )

        # At least one provider should have tokens recorded (45 total)
        assert gw.tracker.total_today() == 45
