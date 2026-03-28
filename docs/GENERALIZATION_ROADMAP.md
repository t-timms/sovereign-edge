# Generalization Roadmap

This document tracks the engineering changes required to transform Sovereign Edge from a single-owner personal system into a general-purpose, multi-user platform.

The current design is intentionally minimal — one owner, one bot token, one device. Each section below describes a specific capability that blocks broader adoption and outlines the concrete changes needed.

---

## 1. Multi-User Authentication

**Current state:** A single `SE_TELEGRAM_OWNER_CHAT_ID` / `SE_DISCORD_OWNER_USER_ID` is hardcoded. Any message from any other user is rejected with `⛔ Unauthorised.`

**What needs to change:**

- Replace the single-owner check with an allowlist stored in a database table (`authorized_users`).
- Add an admin command (e.g., `/adduser <chat_id>`) so the owner can grant access at runtime.
- Introduce per-user conversation history isolation — today all history is keyed by `chat_id`, which already supports this; the bottleneck is the auth layer.
- Optional: role-based access (e.g., `admin` can add users, `user` can only chat, `readonly` cannot trigger morning briefs).

**Files to change:**
- `services/telegram/src/telegram_bot/bot.py` — `_is_owner()` → `_is_authorized()`
- `services/discord/src/discord_bot/bot.py` — same
- `packages/memory/src/memory/conversation.py` — already per-`chat_id`; no structural change needed
- New: `packages/core/src/core/auth.py` — user store backed by SQLite

---

## 2. Web UI / REST API Entry Point

**Current state:** Telegram and Discord are the only interfaces. There is no HTTP API.

**What needs to change:**

- Add a FastAPI service (`services/api/`) that exposes the same `Orchestrator.dispatch()` interface over HTTP.
- `POST /v1/chat` — accepts `{content, chat_id}`, returns `{response, intent, routing, latency_ms}`.
- `GET /v1/health` — expert health status (mirrors `/health` bot command).
- `GET /v1/stats` — today's usage stats (mirrors `/stats` bot command).
- Authentication: Bearer token (`SE_API_TOKEN`) or session cookie for a browser UI.
- Optional: Add a minimal React or HTMX chat UI served from the same FastAPI app.

**Files to add:**
- `services/api/src/api/main.py`
- `services/api/pyproject.toml`
- Entry in `docker-compose.yml`

---

## 3. One-Command Docker Setup

**Current state:** Setup requires: Ollama pull, `uv sync`, `.env` file, optional SOPS setup, manual systemd install. No Docker Compose path exists for first-time users.

**What needs to change:**

- Add a `docker-compose.yml` at the repo root that brings up the full stack: `sovereign-edge`, `ollama`, and optionally `postgres` (if SQLite is replaced — see §5).
- The `sovereign-edge` container should auto-pull the required Ollama models on first start (`ollama pull qwen3:0.6b && ollama pull qwen3-embedding:0.6b`).
- Replace SOPS secrets management with a plain `.env` file path for the Docker use case (SOPS remains available for Jetson/systemd deployments).
- Pin `ollama/ollama` to a specific image tag (not `:latest`) in `docker-compose.yml`.
- Add a `HEALTHCHECK` to the sovereign-edge image.

**Files to add/change:**
- `docker-compose.yml` (new)
- `Dockerfile` (new — multi-stage, non-root user, pinned base)
- `.dockerignore` (new)
- `services/telegram/src/telegram_bot/bot.py` — document env-based startup for Docker

---

## 4. Configurable Expert System

**Current state:** The four experts (Spiritual, Career, Intelligence, Creative) are hardcoded in `_run()` in each bot file. Adding or removing an expert requires code changes.

**What needs to change:**

- Define experts in a config file (e.g., `experts.yaml` or `.env` flag `SE_EXPERTS_ENABLED=spiritual,career,intelligence,creative`).
- The orchestrator reads this at startup and only registers experts that are enabled.
- Each expert's system prompt should be overridable via environment variable or a per-expert config file, without touching Python source.
- The career expert's hardcoded search query in `agents/career/src/career/expert.py` should be fully driven by `SE_CAREER_TARGET_LOCATION` and `SE_CAREER_TARGET_ROLES` (already partially done — verify no hardcoded fallback strings remain).

**Files to change:**
- `agents/orchestrator/src/orchestrator/main.py` — dynamic expert registration
- `agents/career/src/career/expert.py` — remove hardcoded query fallbacks
- `agents/spiritual/src/spiritual/expert.py` — expose denomination/translation preference via env var
- New: `config/experts.yaml` (or env-based approach)

---

## 5. Persistent Storage at Scale

**Current state:** SQLite is used for conversation history and trace storage. `check_same_thread=False` with a `threading.Lock` is safe for a single-user workload. It will not scale to concurrent multi-user access.

**What needs to change:**

- Replace SQLite with PostgreSQL for conversation history and trace storage when deploying for multiple users.
- `packages/memory/src/memory/conversation.py` and `packages/observability/src/observability/traces.py` should accept a database URL via `SE_DATABASE_URL` with SQLite as the default for single-user deployments.
- Use SQLAlchemy async (`asyncpg` driver) to keep the async event loop clean.
- LanceDB (semantic cache) is already file-based and scales independently; no change needed for moderate load.

**Files to change:**
- `packages/memory/src/memory/conversation.py`
- `packages/observability/src/observability/traces.py`
- `packages/core/src/core/config.py` — add `SE_DATABASE_URL: str = "sqlite:///sovereign-edge.db"`

---

## 6. Installable Package / PyPI Distribution

**Current state:** The project is a monorepo with a `uv` workspace. It cannot be installed with `pip install sovereign-edge`.

**What needs to change:**

- Publish the core packages (`core`, `llm`, `router`, `memory`, `search`, `observability`) to PyPI as `sovereign-edge-*`.
- Add a CLI entry point (`se start`, `se health`) via `pyproject.toml` `[project.scripts]`.
- Write a `pip install sovereign-edge && se init` first-run experience that creates `.env.example` and walks the user through required keys.

**Files to change:**
- All `pyproject.toml` files — add PyPI metadata (`authors`, `description`, `classifiers`, `license`)
- New: `packages/cli/` — thin Click/Typer CLI wrapping startup + health check
- CI: add `uv publish` step to GitHub Actions on tag push

---

## 7. Prompt Injection Guard

**Current state:** User input is wrapped in `<user_request>` XML delimiters in expert prompts, which provides weak structural isolation. No dedicated prompt injection classifier is in the pipeline.

**What needs to change:**

- Integrate `meta-llama/Llama-Prompt-Guard-2-86M` (86M param classifier) as a pre-dispatch filter in `orchestrator/main.py`.
- Any message classified as a prompt injection attempt should be rejected before reaching the expert, with a logged CRITICAL event.
- The model is small enough to run on-device via Ollama or directly with `transformers`.

**Files to change:**
- `agents/orchestrator/src/orchestrator/main.py` — add `_check_prompt_injection(text)` before dispatch
- New: `packages/router/src/router/injection_guard.py`

---

## 8. LangGraph State Persistence

**Current state:** Director graph state is in-memory only. If the service restarts mid-conversation (e.g., during a long intelligence brief), state is lost.

**What needs to change:**

- Replace the in-memory `MemorySaver` checkpointer with `AsyncSqliteSaver` from `langgraph-checkpoint-sqlite`.
- Checkpoints persist to `{SE_SSD_ROOT}/langgraph-checkpoints.db`.
- For multi-user / multi-process deployments, replace with `AsyncPostgresSaver`.

**Files to change:**
- Any director graph definition that uses `MemorySaver` — replace with `AsyncSqliteSaver`
- `packages/core/src/core/config.py` — optionally add `SE_LANGGRAPH_CHECKPOINT_PATH`

---

## 9. Eval and Regression Testing

**Current state:** Unit tests cover individual components. There is no golden-set evaluation for expert output quality.

**What needs to change:**

- Add a `tests/evals/` directory with golden-set input/expected-output pairs for each expert.
- Use DeepEval or a simple cosine-similarity check against reference responses to catch quality regressions.
- Run evals in CI on every PR that touches an expert's system prompt or tool configuration.

**Files to add:**
- `tests/evals/test_intelligence_evals.py`
- `tests/evals/test_career_evals.py`
- `tests/evals/test_spiritual_evals.py`
- `tests/evals/test_creative_evals.py`
- `tests/evals/conftest.py` — shared fixtures, golden sets as YAML

---

## Priority Order

For general-purpose adoption, implement in this order:

1. **Docker Compose setup** (§3) — lowest friction for new users; no Jetson required
2. **Multi-user auth** (§1) — gates everything else; single-owner is the primary adoption blocker
3. **Web UI / REST API** (§2) — removes Telegram/Discord account requirement
4. **Configurable experts** (§4) — lets users remove experts they don't need (e.g., Spiritual)
5. **Persistent storage** (§5) — required before production multi-user deployment
6. **Prompt injection guard** (§7) — security baseline for any public-facing deployment
7. **LangGraph persistence** (§8) — reliability improvement
8. **Eval suite** (§9) — quality assurance for ongoing development
9. **PyPI distribution** (§6) — polish; can ship before this
