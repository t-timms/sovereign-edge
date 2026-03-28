"""
structlog configuration — ~5MB RAM, JSON output.

Every log line includes: timestamp, level, expert, model, trace_id.
"""

from __future__ import annotations

import structlog


def setup_logging(debug: bool = False) -> None:
    """Configure structlog for the application."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            10 if debug else 20  # DEBUG=10, INFO=20
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial_context: str) -> structlog.BoundLogger:
    """Get a logger with initial context (e.g., expert name)."""
    return structlog.get_logger(name).bind(**initial_context)


def set_trace_id(trace_id: str) -> None:
    """Bind trace_id to the current async context for all subsequent log entries."""
    structlog.contextvars.bind_contextvars(trace_id=trace_id)


def clear_trace_id() -> None:
    """Remove trace_id binding from the current async context."""
    structlog.contextvars.unbind_contextvars("trace_id")
