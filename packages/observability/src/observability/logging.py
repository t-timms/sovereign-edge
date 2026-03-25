"""
structlog configuration — ~5MB RAM, JSON output.

Every log line includes: timestamp, level, squad, model, trace_id.
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
    """Get a logger with initial context (e.g., squad name)."""
    return structlog.get_logger(name).bind(**initial_context)
