"""Tests for shared Pydantic models in core.types."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from core.types import (
    ExpertName,
    Intent,
    RoutingDecision,
    TaskPriority,
    TaskRequest,
    TaskResult,
)


class TestTaskRequest:
    def test_auto_uuid_on_creation(self) -> None:
        req = TaskRequest(
            content="hello",
            intent=Intent.GENERAL,
            routing=RoutingDecision.CLOUD,
        )
        assert isinstance(req.task_id, UUID)

    def test_two_requests_have_different_ids(self) -> None:
        req1 = TaskRequest(content="a", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        req2 = TaskRequest(content="b", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        assert req1.task_id != req2.task_id

    def test_created_at_is_utc(self) -> None:
        req = TaskRequest(content="x", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        assert req.created_at.tzinfo is not None
        assert req.created_at.tzinfo == UTC or str(req.created_at.tzinfo) == "UTC"

    def test_created_at_is_recent(self) -> None:
        before = datetime.now(UTC)
        req = TaskRequest(content="x", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        after = datetime.now(UTC)
        assert before <= req.created_at <= after

    def test_default_priority_is_normal(self) -> None:
        req = TaskRequest(content="x", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        assert req.priority == TaskPriority.NORMAL

    def test_default_expert_is_general(self) -> None:
        req = TaskRequest(content="x", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        assert req.expert == ExpertName.GENERAL

    def test_default_context_is_empty_dict(self) -> None:
        req = TaskRequest(content="x", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        assert req.context == {}

    def test_default_pii_detected_is_false(self) -> None:
        req = TaskRequest(content="x", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        assert req.pii_detected is False

    def test_context_dicts_are_independent(self) -> None:
        """Two requests must not share the same context dict instance."""
        req1 = TaskRequest(content="a", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        req2 = TaskRequest(content="b", intent=Intent.GENERAL, routing=RoutingDecision.CLOUD)
        req1.context["key"] = "val"
        assert "key" not in req2.context


class TestTaskResult:
    def test_created_at_is_utc(self) -> None:
        from uuid import uuid4

        result = TaskResult(
            task_id=uuid4(),
            expert=ExpertName.INTELLIGENCE,
            content="ok",
            model_used="groq/llama3",
        )
        assert result.created_at.tzinfo is not None

    def test_default_routing_is_cloud(self) -> None:
        from uuid import uuid4

        result = TaskResult(
            task_id=uuid4(),
            expert=ExpertName.CAREER,
            content="ok",
            model_used="groq/llama3",
        )
        assert result.routing == RoutingDecision.CLOUD

    def test_default_numeric_fields_are_zero(self) -> None:
        from uuid import uuid4

        result = TaskResult(
            task_id=uuid4(),
            expert=ExpertName.SPIRITUAL,
            content="ok",
            model_used="none",
        )
        assert result.tokens_in == 0
        assert result.tokens_out == 0
        assert result.latency_ms == 0.0
        assert result.cost_usd == 0.0

    def test_cached_defaults_to_false(self) -> None:
        from uuid import uuid4

        result = TaskResult(
            task_id=uuid4(),
            expert=ExpertName.GENERAL,
            content="ok",
            model_used="none",
        )
        assert result.cached is False
