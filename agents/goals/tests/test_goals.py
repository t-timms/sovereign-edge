"""Tests for goals.store and goals.expert."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from goals.store import GoalStore

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path: Path) -> GoalStore:
    """GoalStore backed by a temp SQLite file."""
    return GoalStore(db_path=tmp_path / "test_goals.db")


# ── GoalStore tests ───────────────────────────────────────────────────────────


def test_add_goal_persists(store: GoalStore) -> None:
    goal_id = store.add_goal("Land ML Engineer role", target_date="2026-06-01")
    goals = store.list_goals()
    assert len(goals) == 1
    assert goals[0].id == goal_id
    assert goals[0].title == "Land ML Engineer role"
    assert goals[0].target_date == "2026-06-01"
    assert goals[0].progress_pct == 0
    assert goals[0].status == "active"


def test_update_progress_stores_correctly(store: GoalStore) -> None:
    gid = store.add_goal("Finish Sovereign Edge v1.0")
    store.update_progress(gid, 75)
    goal = store.get_by_id(gid)
    assert goal is not None
    assert goal.progress_pct == 75


def test_update_progress_clamps_at_100(store: GoalStore) -> None:
    gid = store.add_goal("Overachiever goal")
    store.update_progress(gid, 150)
    assert store.get_by_id(gid).progress_pct == 100  # type: ignore[union-attr]


def test_update_progress_clamps_at_zero(store: GoalStore) -> None:
    gid = store.add_goal("Zero test")
    store.update_progress(gid, -10)
    assert store.get_by_id(gid).progress_pct == 0  # type: ignore[union-attr]


def test_mark_complete_removes_from_active(store: GoalStore) -> None:
    gid = store.add_goal("Complete this goal")
    store.mark_complete(gid)
    active = store.list_goals(status="active")
    assert all(g.id != gid for g in active)
    completed = store.list_goals(status="complete")
    assert any(g.id == gid for g in completed)
    assert store.get_by_id(gid).progress_pct == 100  # type: ignore[union-attr]


def test_get_urgent_returns_max_3(store: GoalStore) -> None:
    for i in range(5):
        store.add_goal(f"Goal {i}", target_date=f"2026-0{i + 1}-01")
    urgent = store.get_urgent(limit=3)
    assert len(urgent) == 3


def test_get_urgent_date_ordered(store: GoalStore) -> None:
    store.add_goal("Far goal", target_date="2027-12-01")
    store.add_goal("Near goal", target_date="2026-05-01")
    urgent = store.get_urgent()
    assert urgent[0].target_date == "2026-05-01"


def test_get_urgent_no_date_last(store: GoalStore) -> None:
    store.add_goal("No date goal")
    store.add_goal("Has date goal", target_date="2026-06-01")
    urgent = store.get_urgent()
    assert urgent[0].target_date is not None


# ── GoalExpert morning_brief tests ────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_morning_brief_empty_when_no_goals(tmp_path: Path) -> None:
    from goals.expert import GoalExpert

    expert = GoalExpert()
    with patch("goals.expert.GoalStore") as mock_store_cls:
        mock_store_cls.return_value.get_urgent.return_value = []
        result = await expert.morning_brief()
    assert result == ""


@pytest.mark.asyncio()
async def test_morning_brief_formats_goals(tmp_path: Path) -> None:
    from goals.expert import GoalExpert
    from goals.store import GoalRecord

    fake_goal = GoalRecord(
        id=1,
        title="Land ML job",
        description="",
        target_date="2026-06-01",
        progress_pct=40,
        status="active",
        created_at="2026-04-04T00:00:00+00:00",
        updated_at="2026-04-04T00:00:00+00:00",
    )

    mock_gateway = MagicMock()
    mock_gateway.complete = AsyncMock(
        return_value={
            "content": "Apply to 3 jobs on LinkedIn today.",
            "model": "test",
            "tokens_in": 10,
            "tokens_out": 10,
            "cost_usd": 0.0,
        }
    )

    expert = GoalExpert()
    with (
        patch("goals.expert.GoalStore") as mock_store_cls,
        patch("llm.gateway.get_gateway", return_value=mock_gateway),
    ):
        mock_store_cls.return_value.get_urgent.return_value = [fake_goal]
        result = await expert.morning_brief()

    assert "Land ML job" in result
    assert "40%" in result
    assert "2026-06-01" in result
    assert "Action" in result
