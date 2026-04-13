"""Tests for the agent audit trail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from observability.audit import AuditEventType, AuditStore, _hash_content


class TestHashContent:
    def test_consistent_hash(self) -> None:
        assert _hash_content("hello") == _hash_content("hello")

    def test_different_inputs_different_hashes(self) -> None:
        assert _hash_content("hello") != _hash_content("world")

    def test_hash_length(self) -> None:
        assert len(_hash_content("test")) == 16


class TestAuditStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> AuditStore:
        s = AuditStore(tmp_path / "test_audit.db")
        yield s
        s.close()

    def test_record_dispatch(self, store: AuditStore) -> None:
        store.record(
            event_type=AuditEventType.DISPATCH,
            expert="intelligence",
            task_id="task-001",
            model="groq/llama-4-scout",
            input_text="What is LoRA?",
            output_text="LoRA is a parameter-efficient...",
            tokens_in=50,
            tokens_out=100,
            latency_ms=200.0,
        )
        events = store.get_events(event_type="dispatch")
        assert len(events) == 1
        assert events[0]["expert"] == "intelligence"
        # Verify content is hashed, not raw
        assert events[0]["input_hash"] != "What is LoRA?"
        assert len(events[0]["input_hash"]) == 16

    def test_record_tool_call(self, store: AuditStore) -> None:
        store.record(
            event_type=AuditEventType.TOOL_CALL,
            expert="intelligence",
            latency_ms=50.0,
            metadata={"tool_name": "arxiv_search", "query": "LoRA fine-tuning"},
        )
        events = store.get_events(event_type="tool_call")
        assert len(events) == 1

    def test_record_error(self, store: AuditStore) -> None:
        store.record(
            event_type=AuditEventType.ERROR,
            expert="career",
            status="error",
            metadata={"error": "timeout"},
        )
        errors = store.get_error_events()
        assert len(errors) == 1
        assert errors[0]["status"] == "error"

    def test_dispatch_summary(self, store: AuditStore) -> None:
        for expert in ["intelligence", "intelligence", "spiritual"]:
            store.record(
                event_type=AuditEventType.DISPATCH,
                expert=expert,
                latency_ms=100.0,
                tokens_in=50,
                tokens_out=80,
            )
        summary = store.get_dispatch_summary()
        experts = {e["expert"]: e for e in summary["experts"]}
        assert experts["intelligence"]["dispatches"] == 2
        assert experts["spiritual"]["dispatches"] == 1

    def test_daily_report(self, store: AuditStore) -> None:
        store.record(event_type=AuditEventType.DISPATCH, expert="intel")
        store.record(event_type=AuditEventType.CACHE_HIT, expert="intel")
        store.record(event_type=AuditEventType.TOOL_CALL, expert="intel")
        report = store.get_daily_audit_report()
        assert report["total_events"] == 3
        assert len(report["by_type"]) == 3

    def test_empty_store(self, store: AuditStore) -> None:
        events = store.get_events()
        assert events == []
        report = store.get_daily_audit_report()
        assert report["total_events"] == 0

    def test_filter_by_expert(self, store: AuditStore) -> None:
        store.record(event_type=AuditEventType.DISPATCH, expert="a")
        store.record(event_type=AuditEventType.DISPATCH, expert="b")
        events = store.get_events(expert="a")
        assert len(events) == 1
        assert events[0]["expert"] == "a"

    def test_metadata_stored_as_json(self, store: AuditStore) -> None:
        store.record(
            event_type=AuditEventType.TOOL_CALL,
            metadata={"tool_name": "bible_api", "ref": "John 3:16"},
        )
        events = store.get_events()
        meta = json.loads(events[0]["metadata"])
        assert meta["tool_name"] == "bible_api"
