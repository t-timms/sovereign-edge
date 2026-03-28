"""Base interface that all agent experts must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from core.types import TaskRequest, TaskResult


class BaseExpert(ABC):
    """Interface that all experts must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Expert identifier (e.g., 'spiritual')."""
        ...

    @abstractmethod
    async def process(self, task: TaskRequest) -> TaskResult:
        """Process a task and return a result."""
        ...

    @abstractmethod
    async def morning_brief(self) -> str:
        """Generate content for the morning digest. Called at 05:00 CT."""
        ...

    async def stream_process(self, task: TaskRequest) -> AsyncGenerator[str, None]:
        """Stream response chunks. Default: yields full process() result at once.

        Override in experts that support true token-by-token streaming to give
        users live feedback as the LLM generates rather than waiting for completion.
        """
        result = await self.process(task)
        yield result.content

    async def health_check(self) -> bool:
        """Return True if expert is operational."""
        return True
