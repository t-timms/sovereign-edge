# ============================================================
# Stage 1 — Builder
# Install uv, sync the full workspace (no dev deps)
# ============================================================
FROM python:3.13.2-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /usr/local/bin/

# Build deps (curl needed for uv self-update + healthcheck later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace manifests first for layer caching
COPY pyproject.toml uv.lock* ./
COPY packages/core/pyproject.toml         packages/core/pyproject.toml
COPY packages/llm/pyproject.toml          packages/llm/pyproject.toml
COPY packages/memory/pyproject.toml       packages/memory/pyproject.toml
COPY packages/observability/pyproject.toml packages/observability/pyproject.toml
COPY packages/router/pyproject.toml       packages/router/pyproject.toml
COPY packages/search/pyproject.toml       packages/search/pyproject.toml
COPY agents/career/pyproject.toml         agents/career/pyproject.toml
COPY agents/creative/pyproject.toml       agents/creative/pyproject.toml
COPY agents/director/pyproject.toml       agents/director/pyproject.toml
COPY agents/goals/pyproject.toml          agents/goals/pyproject.toml
COPY agents/intelligence/pyproject.toml   agents/intelligence/pyproject.toml
COPY agents/orchestrator/pyproject.toml   agents/orchestrator/pyproject.toml
COPY agents/spiritual/pyproject.toml      agents/spiritual/pyproject.toml
COPY services/discord/pyproject.toml      services/discord/pyproject.toml
COPY services/health/pyproject.toml       services/health/pyproject.toml
COPY services/mcp/pyproject.toml          services/mcp/pyproject.toml
COPY services/telegram/pyproject.toml     services/telegram/pyproject.toml
COPY services/whatsapp/pyproject.toml     services/whatsapp/pyproject.toml

# Copy all source (needed by hatchling editable installs)
COPY packages/ packages/
COPY agents/   agents/
COPY services/ services/

# Sync: install all workspace packages + deps, no dev extras
ENV UV_SYSTEM_PYTHON=0
RUN uv sync --no-dev --all-packages

# ============================================================
# Stage 2 — Runtime
# Lean image: only .venv + src trees, no build tools
# ============================================================
FROM python:3.13.2-slim AS runtime

LABEL org.opencontainers.image.title="sovereign-edge" \
      org.opencontainers.image.version="0.3.1" \
      org.opencontainers.image.source="https://github.com/t-timms/sovereign-edge"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --uid 1001 --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the virtualenv produced by the builder
COPY --from=builder /build/.venv /app/.venv

# Copy source trees (hatchling editable installs point into these)
COPY --from=builder /build/packages  /app/packages
COPY --from=builder /build/agents    /app/agents
COPY --from=builder /build/services  /app/services

# Data directory (SQLite traces, LanceDB, logs) — mounted as volume at runtime
RUN mkdir -p /app/data && chown appuser:appuser /app/data

# Entrypoint script: starts health server in background, then runs the bot.
# Install before USER so root can set permissions.
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && chown appuser:appuser /app/entrypoint.sh

# Ensure .env is never baked into the image
# (reminder: always pass secrets via --env-file / docker-compose env_file)

USER appuser

# Activate venv for all subsequent RUN / CMD / ENTRYPOINT invocations
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Point settings defaults at /app/data instead of the Jetson path
    SE_SSD_ROOT=/app/data \
    SE_LANCEDB_PATH=/app/data/lancedb \
    SE_LOGS_PATH=/app/data/logs \
    SE_MODELS_PATH=/app/data/models \
    SE_PROJECT_ROOT=/app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
