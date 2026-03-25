"""Shared types used across all packages and agents."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Intent(StrEnum):
    """Intent classes for the ONNX router."""

    SPIRITUAL = "SPIRITUAL"
    CAREER = "CAREER"
    INTELLIGENCE = "INTELLIGENCE"
    CREATIVE = "CREATIVE"
    GENERAL = "GENERAL"


class SquadName(StrEnum):
    """Agent squad identifiers."""

    SPIRITUAL = "spiritual"
    CAREER = "career"
    INTELLIGENCE = "intelligence"
    CREATIVE = "creative"
    GENERAL = "general"
    ORCHESTRATOR = "orchestrator"


class TaskPriority(StrEnum):
    """Priority levels for the inference queue."""

    CRITICAL = "P0"  # HITL response, security alert
    HIGH = "P1"      # Director orchestration, morning digest
    NORMAL = "P2"    # Squad worker tasks
    LOW = "P3"       # Content generation, non-urgent


class RoutingDecision(StrEnum):
    """Where to process a request."""

    LOCAL = "LOCAL"  # PII detected — process on Jetson only
    CLOUD = "CLOUD"  # Route to free cloud APIs via LiteLLM
    CACHE = "CACHE"  # Semantic cache hit in LanceDB


class TaskRequest(BaseModel):
    """A task submitted to any squad."""

    task_id: UUID = Field(default_factory=uuid4)
    content: str
    intent: Intent
    routing: RoutingDecision
    priority: TaskPriority = TaskPriority.NORMAL
    squad: SquadName = SquadName.GENERAL
    context: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    pii_detected: bool = False


class TaskResult(BaseModel):
    """Result from a squad's processing."""

    task_id: UUID
    squad: SquadName
    content: str
    model_used: str
    routing: RoutingDecision = RoutingDecision.CLOUD
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, str] = Field(default_factory=dict)
