"""
structlog configuration — ~5MB RAM, JSON output.

Every log line includes: timestamp, level, expert, model, trace_id.
Also bridges stdlib logging.getLogger() so all modules (search, career, etc.)
output to the same stream visible in journalctl.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(debug: bool = False) -> None:
    """Configure structlog + stdlib logging for the application.

    structlog loggers (via get_logger) produce JSON directly.
    stdlib loggers (via logging.getLogger) are bridged through structlog
    processors so their output is also JSON-formatted and visible in journalctl.
    """
    log_level = logging.DEBUG if debug else logging.INFO

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # ── structlog loggers ─────────────────────────────────────────────────
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            10 if debug else 20  # DEBUG=10, INFO=20
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ── stdlib logging bridge ─────────────────────────────────────────────
    # Modules that use logging.getLogger(__name__) (search.jobs, job_store,
    # career subgraph, etc.) were silently dropped because no handler existed.
    # This bridges them through structlog's JSON formatter.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def get_logger(name: str, **initial_context: str) -> structlog.BoundLogger:
    """Get a logger with initial context (e.g., expert name)."""
    return structlog.get_logger(name).bind(**initial_context)


def set_trace_id(trace_id: str) -> None:
    """Bind trace_id to the current async context for all subsequent log entries."""
    structlog.contextvars.bind_contextvars(trace_id=trace_id)


def clear_trace_id() -> None:
    """Remove trace_id binding from the current async context."""
    structlog.contextvars.unbind_contextvars("trace_id")
