# Sovereign Edge — Architecture Reference

> **Audience:** Engineers onboarding to the codebase, contributors evaluating design decisions, and the author revisiting choices months later.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Component Architecture](#3-component-architecture)
4. [Request Lifecycle](#4-request-lifecycle)
5. [LLM Gateway](#5-llm-gateway)
6. [Memory Architecture](#6-memory-architecture)
7. [Human-in-the-Loop (HITL)](#7-human-in-the-loop-hitl)
8. [Security Model](#8-security-model)
9. [Observability](#9-observability)
10. [Voice Pipeline](#10-voice-pipeline)
11. [Scheduling](#11-scheduling)
12. [Configuration Reference](#12-configuration-reference)
13. [Deployment](#13-deployment)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Testing Strategy](#15-testing-strategy)

---

## 1. System Overview

Sovereign Edge is a privacy-first, always-on personal intelligence system designed to run on a **Jetson Orin Nano 8GB** (~$2–3/month electricity). It classifies user intent in under 10ms via an ONNX-quantized DistilBERT router, dispatches to specialized domain squads, and delegates generation to free-tier cloud LLMs through a unified failover gateway.

```
JETSON ORIN NANO 8GB

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Input Layer                                                         │
  │  Telegram Bot  ──or──  Voice Pipeline (OpenWakeWord → Whisper)       │
  └────────────────────────────┬─────────────────────────────────────────┘
                               │ user_text
                               ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  ONNX Intent Router (DistilBERT INT8, <10ms)                        │
  │  4-class: SPIRITUAL | CAREER | INTELLIGENCE | CREATIVE              │
  └────────────────────────────┬─────────────────────────────────────────┘
                               │ intent + confidence
                               ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  LangGraph StateGraph Orchestrator                                   │
  │                                                                      │
  │  router_node ──(4 conditional edges)──▶ squad_node                  │
  │                                              │                       │
  │  memory_node ◀───────────────────────────────┘                       │
  │       │                                                              │
  │  hitl_node  (LangGraph interrupt — Telegram approval gate)           │
  │       │                                                              │
  │  delivery_node                                                       │
  └──────────────────────────────────────────────────────────────────────┘
         │             │              │              │
    Spiritual       Career        Intelligence    Creative
    Squad           Squad         Squad           Squad
    (Bible RAG)     (JobSpy +     (AlphaVantage   (Manim /
                    Jina)         / arXiv / FMP)   FFmpeg /
                                                   Kokoro)
         │
  Shared: LanceDB · Mem0 · structlog · SQLite · APScheduler

         │ (generation via LiteLLM — free tiers only)
  Groq → Gemini Flash → Cerebras → Mistral  (priority order, failover)
```

**Key numbers:**
- Intent classification: <10ms (ONNX INT8 on Jetson CPU)
- LLM timeout: 30s hard ceiling per provider attempt
- LLM retries: 3 attempts with exponential backoff (2s → 30s) on rate-limit/timeout
- Memory retrieval: Mem0 episodic + LanceDB semantic + SQLite skill patterns
- Peak RAM: ~5.1 GB on Jetson (headroom for LLM spikes)

---

## 2. Repository Layout

```
sovereign-edge/
├── packages/                   # Shared libraries (no business logic)
│   ├── core/                   # Config, exceptions, types, security utils
│   ├── llm/                    # LiteLLM gateway — provider failover + retry
│   ├── memory/                 # Mem0 store + SQLite skill library
│   ├── observability/          # structlog setup + OTEL tracing decorator
│   └── router/                 # ONNX DistilBERT intent classifier
│
├── agents/                     # Domain specialists (pure business logic)
│   ├── orchestrator/           # LangGraph StateGraph + HITL + graph entry
│   ├── spiritual/              # Bible RAG, devotionals, theological Q&A
│   ├── career/                 # Job scraping, resume tailoring
│   ├── intelligence/           # Market data, arXiv, earnings, FMP
│   └── creative/               # Script writing, diagram generation
│
├── services/                   # Delivery channels (I/O boundaries)
│   ├── telegram/               # PTB bot, command handlers, APScheduler
│   └── voice/                  # Wake word, STT, TTS, voice pipeline
│
├── scripts/                    # One-shot utilities (data ingest, training)
├── evals/                      # LLM evaluation harness + judges
├── data/                       # Static data (Bible datasets, ONNX model)
├── systemd/                    # Service unit files for Jetson boot
├── secrets/                    # SOPS-encrypted env vars (safe to commit)
├── pyproject.toml              # Workspace root — uv workspaces
├── uv.lock                     # Committed lockfile for reproducible builds
└── Taskfile.yml                # Task runner (start, stop, lint, test, etc.)
```

### Dependency graph (no cycles)

```
services/telegram  ──▶  agents/orchestrator  ──▶  packages/llm
services/voice     ──▶  agents/orchestrator  ──▶  packages/memory
                   ──▶  agents/spiritual     ──▶  packages/core
                   ──▶  agents/career        ──▶  packages/router
                   ──▶  agents/intelligence  ──▶  packages/observability
                   ──▶  agents/creative
```

All agents depend only on `packages/`. Services depend on agents and packages. No package depends on an agent or service.

---

## 3. Component Architecture

### 3.1 `packages/core`

Foundation layer. Everything else imports from here.

| Module | Purpose |
|--------|---------|
| `config.py` | `Settings` (pydantic-settings, `.env` → typed fields). `get_settings()` is `@lru_cache(maxsize=1)` — singleton per process. |
| `exceptions.py` | `LLMError`, `ConfigurationError`, `AuthError` — typed exceptions for clean catch sites |
| `types.py` | `IntentClass` enum (`SPIRITUAL`, `CAREER`, `INTELLIGENCE`, `CREATIVE`) |
| `security.py` | `sanitize_input()` — regex allowlist strips prompt injection patterns |
| `logging.py` | structlog processor chain (plain dev / JSON prod) |

### 3.2 `packages/llm`

Unified LLM gateway. All generation calls go through `LLMGateway`.

See [§5 LLM Gateway](#5-llm-gateway) for full details.

### 3.3 `packages/memory`

Three-tier memory:

| Tier | Implementation | Purpose |
|------|---------------|---------|
| Episodic | Mem0 (`memory/store.py`) | "You mentioned X last week" — fuzzy semantic recall |
| Semantic | LanceDB (`spiritual/bible_rag.py`) | Scripture chunk retrieval via embeddings |
| Procedural | SQLite (`memory/skill_library.py`) | Skill pattern reinforcement by intent class |

### 3.4 `packages/router`

`IntentRouter` — loads a DistilBERT INT8 ONNX model and returns `(IntentClass, confidence_float)` in <10ms. Threshold: 0.7. Below threshold defaults to `INTELLIGENCE`.

### 3.5 `packages/observability`

- `setup_observability(settings)` — configures structlog + optional OTEL exporter
- `@traced("span_name")` decorator — wraps async functions with OTEL spans and binds a `correlation_id` UUID to structlog contextvars for the duration of the call. Nested `@traced` calls detect the existing correlation_id and do not overwrite it.

### 3.6 `agents/orchestrator`

The control plane. See [§4 Request Lifecycle](#4-request-lifecycle).

**Key files:**
- `graph.py` — all LangGraph node implementations + `run_turn()` + `resume_turn()`
- `state.py` — `SovereignState` TypedDict definition

### 3.7 Agent Squads

Each squad exposes a single `async def run(state: SovereignState) -> str` interface. The orchestrator calls this and never touches squad internals.

| Squad | HITL required | Key tools |
|-------|--------------|-----------|
| Spiritual | No | LanceDB Bible RAG, Mem0 cross-reference |
| Career | **Yes** (apply/email actions) | JobSpy, Jina reranker, python-docx |
| Intelligence | No | yFinance, Alpha Vantage, arXiv API, FMP transcripts |
| Creative | **Yes** (publish/post actions) | Manim, FFmpeg, Kokoro TTS, D2 |

### 3.8 `services/telegram`

- `bot.py` — PTB `Application`, command/message/callback handlers, HITL inline keyboards
- `scheduler.py` — APScheduler jobs: morning brief (6am), job scan (9am), market summary (6pm)
- Per-user rate limiting: `_RATE_LIMIT_SECONDS = 2.0` — prevents runaway message storms
- Auth: single-user whitelist via `TELEGRAM_ALLOWED_USER_ID`

### 3.9 `services/voice`

Optional local voice interface. All voice deps are optional extras — base package works without them.

```
OpenWakeWord  →  faster-whisper  →  orchestrator.run_turn()  →  Kokoro/Piper TTS
(wake.py)         (stt.py)            (pipeline.py)               (tts.py)
```

Install extras: `pip install sovereign-edge-voice[full]`

---

## 4. Request Lifecycle

### 4.1 Happy path (no HITL)

```
User message (Telegram)
  │
  ├─ Auth check (telegram_allowed_user_id)
  ├─ Rate limit check (2s minimum gap)
  │
  ▼
run_turn(user_text, thread_id=f"user_{user_id}")
  │
  ├─ sanitize_input(user_text)           ← strips injection patterns
  ├─ IntentRouter.classify(text)         ← ONNX, <10ms
  ├─ MemoryStore.format_context(text)    ← Mem0 episodic recall (≤6000 chars)
  ├─ SkillLibrary.get_top_skills(intent) ← SQLite procedural memory
  │
  ├─ squad_node (spiritual|career|intelligence|creative)
  │     └─ LLMGateway.complete(messages) ← with provider failover + retry
  │
  ├─ memory_node                         ← persist result to Mem0
  │     └─ SkillLibrary.record_outcome() ← reinforce skill patterns
  │
  ├─ delivery_node                       ← package final message
  │
  └─ return response_text
         │
         ▼
  msg.reply_text(result, parse_mode="Markdown")
```

### 4.2 HITL path (career/creative actions)

```
run_turn()
  │
  ├─ career_node or creative_node  → sets hitl_required=True
  │
  ├─ memory_node
  │
  ├─ hitl_node  ← LangGraph interrupt() — graph is SUSPENDED here
  │     └─ stores thread_id in _PENDING_HITL dict
  │
  └─ returns "" (empty string signals HITL pending to telegram bot)
         │
         ▼
  bot.py displays InlineKeyboard [Approve] [Reject]
  User taps Approve → hitl_callback() → resume_turn(thread_id, approved=True)
         │
         ▼
  LangGraph resumes at hitl_node → delivery_node → response sent
```

### 4.3 Scheduled (proactive) path

APScheduler jobs call `run_turn()` directly with a `schedule_trigger` kwarg. The resulting message is sent to `telegram_allowed_user_id` via `bot.send_message()`. No user interaction initiates these turns.

---

## 5. LLM Gateway

**File:** `packages/llm/src/llm/gateway.py`

### Provider priority

```
Groq (llama-3.3-70b-versatile)
  → Gemini Flash (gemini-1.5-flash)
    → Cerebras (llama3.1-70b)
      → Mistral (mistral-large-latest)
```

Only providers with a non-empty API key are included. `active_llm_providers()` in `Settings` builds the list at runtime.

### Retry and failover

```
For each provider:
  ├─ @retry(RateLimitError | Timeout, exponential backoff 2s→30s, max 3 attempts)
  ├─ Hard timeout: 30s per attempt (request_timeout kwarg)
  └─ On any non-LLMError exception → log warning, continue to next provider

If all providers fail → raise LLMError("All LLM providers failed")
```

### Methods

| Method | Use |
|--------|-----|
| `complete(messages, *, system, max_tokens, temperature)` | Standard single response |
| `stream(messages, *, system, max_tokens, temperature)` | `AsyncGenerator[str]` token stream with provider fallback |
| `complete_structured(messages, *, schema, system, max_tokens)` | JSON response validated against a Pydantic model (temperature=0.2) |

### `complete_structured` pattern

```python
class EarningsSummary(BaseModel):
    symbol: str
    verdict: str
    key_points: list[str]

summary = await gateway.complete_structured(
    [Message.user(transcript_text)],
    schema=EarningsSummary,
    system="Summarize this earnings call in JSON.",
)
# summary is a validated EarningsSummary instance — never raw JSON
```

Raises `LLMError` if the LLM returns malformed JSON or a schema mismatch. Never raises `ValidationError` to callers.

---

## 6. Memory Architecture

### Tier 1 — Episodic (Mem0 + LanceDB)

`packages/memory/src/memory/store.py`

- Backed by Mem0 with LanceDB as the vector store
- `format_context(query)` → semantic search over past interactions → formatted string for LLM prompt
- Context is capped at **6,000 characters** in `router_node` before injection. The tail is kept (most recent context wins).
- `add_memory(text)` is called in `memory_node` after every successful turn

### Tier 2 — Semantic (LanceDB Bible RAG)

`agents/spiritual/src/spiritual/bible_rag.py`

- STEPBible datasets pre-indexed into LanceDB (`data/lancedb/`)
- Verse lookup uses an allowlist regex (`_REF_SAFE = re.compile(r"^[A-Za-z0-9 .:]+$")`) before any WHERE clause construction — prevents injection into LanceDB query strings
- Embedding via `nomic-embed-text` (Ollama)

### Tier 3 — Procedural (SQLite skill patterns)

`packages/memory/src/memory/skill_library.py`

- Records per-intent success/failure outcomes
- `get_top_skills(intent, limit=2)` returns the most effective recent patterns as prompt context
- Updated in `memory_node` (automatic success) and `resume_turn` (HITL outcome)

### Singleton management

Both `MemoryStore` and `SkillLibrary` are instantiated once per process via `@functools.lru_cache(maxsize=1)` factory functions in `graph.py`. This prevents per-request DB connection churn across async handler calls.

---

## 7. Human-in-the-Loop (HITL)

HITL uses LangGraph's native `interrupt()` mechanism — not polling, not a callback queue.

### How it works

1. `career_node` or `creative_node` sets `hitl_required=True` in graph state
2. `hitl_node` calls `interrupt(payload)` — this is a LangGraph primitive that serializes the graph state and raises a special exception caught by LangGraph's executor
3. Graph execution is **suspended** — the compiled graph holds the checkpoint in `MemorySaver` keyed by `thread_id`
4. `run_turn()` returns `""` (empty string) — the Telegram bot detects this and displays the approval keyboard
5. `_PENDING_HITL[str(user_id)] = thread_id` records the pending approval
6. User taps Approve/Reject → `hitl_callback()` → `_PENDING_HITL.pop()` (atomic dict operation, safe in single-threaded asyncio) → `resume_turn(thread_id, approved=bool)`
7. `resume_turn()` calls `_compiled.ainvoke({"hitl_approved": approved}, config)` — LangGraph resumes from the interrupt point

### Double-fire prevention

`_PENDING_HITL.pop(user_id, None)` returns `None` if already handled (button tapped twice, or `/approve` command sent concurrently). The `None` check in both `hitl_callback` and `cmd_approve/reject` ensures exactly one `resume_turn()` call per pending action.

---

## 8. Security Model

### 8.1 Authentication

Single-user whitelist: `TELEGRAM_ALLOWED_USER_ID` (integer Telegram user ID). Every handler checks `_check_auth(user_id, settings)` before processing. Unauthenticated requests receive "Unauthorized." and are dropped.

### 8.2 Prompt injection defense

Defense-in-depth — three layers:

| Layer | Location | Mechanism |
|-------|----------|-----------|
| Input sanitization | `core/security.py` → `router_node` | Regex strips 10+ injection patterns (`ignore previous instructions`, `<system>`, `[INST]`, etc.) |
| Role separation | All LLM calls | User content and system prompt are always in separate message roles — never concatenated |
| Output validation | `LLMGateway.complete_structured` | Pydantic schema enforces expected output shape before consumption |

### 8.3 Database injection

`bible_rag.py::lookup_verse()` validates the reference string against `_REF_SAFE = re.compile(r"^[A-Za-z0-9 .:]+$")` before any WHERE clause construction. Invalid references return `None` and log a warning.

### 8.4 Secrets management

- All secrets in `.env` (gitignored) or SOPS-encrypted `secrets/env.enc.yaml`
- `pydantic-settings` reads from env — no secret ever appears in source code
- `.env.example` documents all variables with defaults

### 8.5 Rate limiting

Per-user: `_RATE_LIMIT_SECONDS = 2.0` minimum gap enforced with `time.monotonic()`. Prevents runaway bot abuse or accidental message storms from the single authorized user.

### 8.6 Logging discipline

- Error messages log `error_type=type(exc).__name__` only — never raw exception strings that might contain tokens, API keys, or internal paths
- `exc_info=True` used for full tracebacks at `logger.error()` level only (not warnings)

---

## 9. Observability

### Structured logging

`packages/observability/src/observability/setup.py`

- Dev: plain-text via structlog's `dev_output` renderer
- Prod (`LOG_JSON=true`): JSON lines via `structlog.processors.JSONRenderer`
- Log level: `LOG_LEVEL` env var (default `INFO`), validated to allowed set

All log calls use `logger.info("event.name", key=value)` — never f-strings in log calls (lazy evaluation).

### OTEL tracing

`packages/observability/src/observability/tracing.py`

- `@traced("span_name")` wraps async functions with OTEL spans
- Binds a `correlation_id` UUID to structlog contextvars on entry
- Nested `@traced` calls detect an existing `correlation_id` and do not overwrite it — all spans within a request share the same ID
- OTEL exporter: `OTEL_ENDPOINT` env var (OTLP gRPC). Empty = no export (no-op spans only)

### Key log events

| Event | Level | Location |
|-------|-------|----------|
| `orchestrator.routed` | INFO | graph.py — intent + confidence per turn |
| `llm.complete` | INFO | gateway.py — model used + total tokens |
| `llm.provider_failed` | WARNING | gateway.py — provider name + error type |
| `telegram.message` | INFO | bot.py — user_id + message length |
| `telegram.hitl_callback.invalid_data` | WARNING | bot.py — callback data length |
| `voice.stt.transcribed` | INFO | stt.py — text length + language probability |
| `voice.tts.done` | INFO | tts.py — engine used |

---

## 10. Voice Pipeline

**File:** `services/voice/src/voice/pipeline.py`

All voice dependencies are optional extras. The base package installs without any audio libraries.

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐    ┌───────────────┐
│ OpenWakeWord│───▶│ faster-      │───▶│ orchestrator.        │───▶│ Kokoro-82M    │
│ (wake.py)   │    │ whisper      │    │ run_turn()           │    │ or Piper TTS  │
│ ONNX model  │    │ (stt.py)     │    │ (full LangGraph       │    │ (tts.py)      │
│ "hey        │    │ VAD + beam=5 │    │  pipeline)           │    │ sounddevice   │
│  sovereign" │    │ 16kHz int16  │    │                      │    │ playback      │
└─────────────┘    └──────────────┘    └──────────────────────┘    └───────────────┘
```

### STT (stt.py)

- Model: `faster-whisper` — CTranslate2 format, significantly faster than HuggingFace Whisper
- Model size from `settings.stt_model` (default `base.en`, override in `.env`: `STT_MODEL=small.en`)
- Device: auto-detected (CUDA if `torch` available, else CPU)
- Compute: `int8` on CPU, `float16` on CUDA
- VAD filter enabled — skips silence segments

### TTS (tts.py)

- Primary: Kokoro-82M (`pip install kokoro`) — 96x real-time, MOS 4.1
- Fallback: Piper TTS (`pip install piper-tts`) — 50ms latency, MOS 3.5
- Both return raw PCM int16 bytes + sample_rate for sounddevice playback

### Pipeline model selection

`VoicePipeline` accepts `whisper_model: str | None = None`. When `None`, it reads `self._settings.stt_model` — so `.env` `STT_MODEL` overrides propagate consistently without caller changes.

---

## 11. Scheduling

**File:** `services/telegram/src/telegram_bot/scheduler.py`

APScheduler `BackgroundScheduler` with three jobs:

| Job | Default hour | `Settings` field | What it runs |
|-----|-------------|-----------------|-------------|
| Morning brief | 6am | `morning_brief_hour` | Bible verse + market pre-open + top job leads |
| Job scan | 9am | `job_scan_hour` | Multi-board job scrape filtered to DFW + target roles |
| Market summary | 6pm | `market_summary_hour` | Watchlist alerts + earnings context + arXiv digest |

All jobs call `run_turn()` with `schedule_trigger=<job_id>` — no special code paths, same graph as interactive turns.

Scheduler shutdown is wired to the OS signal handler in `bot.py`: `scheduler.shutdown(wait=False)` followed by `os._exit(0)`.

---

## 12. Configuration Reference

All settings come from environment variables (`.env` file or host environment). See `.env.example` for a full annotated reference.

| Variable | Type | Default | Required |
|----------|------|---------|---------|
| `GROQ_API_KEY` | str | `""` | One of the four LLM keys is required |
| `GEMINI_API_KEY` | str | `""` | — |
| `CEREBRAS_API_KEY` | str | `""` | — |
| `MISTRAL_API_KEY` | str | `""` | — |
| `TELEGRAM_BOT_TOKEN` | str | `""` | **Yes** |
| `TELEGRAM_ALLOWED_USER_ID` | int | `0` | **Yes** |
| `FMP_API_KEY` | str | `""` | No — enables earnings transcript summaries |
| `STT_MODEL` | str | `base.en` | No — `tiny.en` / `base.en` / `small.en` / `medium.en` |
| `WATCHLIST` | list[str] | `NVDA,MSFT,GOOGL,META` | No |
| `LOG_JSON` | bool | `false` | No — set `true` in production |
| `LOG_LEVEL` | str | `INFO` | No |
| `OTEL_ENDPOINT` | str | `""` | No — OTLP gRPC endpoint |
| `MORNING_BRIEF_HOUR` | int | `6` | No — 0–23 |
| `JOB_SCAN_HOUR` | int | `9` | No |
| `MARKET_SUMMARY_HOUR` | int | `18` | No |
| `LANCEDB_PATH` | Path | `data/lancedb` | No |
| `SKILL_DB_PATH` | Path | `data/skills.db` | No |
| `MEM0_USER_ID` | str | `john` | No |

---

## 13. Deployment

### Jetson Orin Nano (primary target)

```bash
# 1. Flash JetPack 6, enable SSH

# 2. Bootstrap system deps
bash scripts/setup-jetson.sh   # Docker, uv, ONNX runtime, Ollama

# 3. Clone and install
git clone <repo> ~/sovereign-edge && cd ~/sovereign-edge
uv sync

# 4. Configure secrets (SOPS + age)
age-keygen -o ~/.config/sops/age/keys.txt
# Add public key to secrets/.sops.yaml
sops -e -i secrets/env.enc.yaml   # fill in real API keys

# 5. Install systemd services
sudo cp systemd/*.service systemd/*.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-stack.target

# 6. Verify
task status
task logs
```

### Systemd services

| Unit | Manages |
|------|---------|
| `sovereign-telegram.service` | Telegram bot + APScheduler |
| `sovereign-voice.service` | Voice pipeline (optional) |
| `ai-stack.target` | Group target — enables/stops all services together |

All services are configured with `Restart=on-failure` and `RestartSec=10`.

### Task runner shortcuts

```bash
task start          # Start ai-stack.target
task stop           # Stop ai-stack.target
task status         # Show service health
task logs           # Tail all logs (journalctl)
task monitor        # jtop resource monitor
task lint           # ruff check . --fix && ruff format .
task test           # pytest
task eval           # Evaluation harness
task train-router   # Retrain ONNX intent router
task secrets-edit   # sops -e secrets/env.enc.yaml
```

### Memory budget (Jetson Orin Nano 8GB)

| Component | RAM |
|-----------|-----|
| JetPack 6 OS (headless) | ~1.2 GB |
| LangGraph orchestrator + Python runtime | ~150–200 MB |
| ONNX Router (DistilBERT INT8) | ~200 MB |
| LiteLLM library | ~75 MB |
| Ollama (nomic-embed-text only) | ~400 MB |
| LanceDB + Mem0 | ~300 MB |
| **Baseline** | **~2.3–2.4 GB** |
| **Peak (content tools active)** | **~4.5–5.1 GB** |
| **Headroom** | **~2.9 GB** |

---

## 14. Key Design Decisions

### Why LangGraph instead of plain async + callbacks?

LangGraph's `StateGraph` gives three things that are hard to build correctly by hand:

1. **Persistent checkpointing** — `MemorySaver` serializes the full graph state between turns, enabling multi-session memory without a custom session store
2. **HITL interrupts** — `interrupt()` is a first-class primitive; the graph suspends cleanly, and `ainvoke()` resumes from exactly the same state after approval
3. **Conditional routing** — intent-based dispatch is a single `add_conditional_edges` call, not a chain of if/else

### Why ONNX INT8 router instead of calling the LLM for routing?

Calling an LLM to classify intent adds ~1–3 seconds and one full API round-trip before work even begins. A DistilBERT INT8 model on the Jetson CPU answers in <10ms, costs nothing, and — critically — never fails due to rate limits. The router is the only inference that runs locally; all generation is delegated to the cloud.

### Why LiteLLM instead of direct OpenAI/Groq SDK calls?

Single interface, four free providers, automatic failover. If Groq rate-limits at 3am, Gemini picks up without any code change. The fallback chain handles ~8–11M tokens/day across all providers combined on free tiers.

### Why LanceDB instead of Chroma/Qdrant?

Embedded: no server process, no port, no Docker container on the Jetson. Persists to disk, ~300 MB footprint, fast enough for single-user personal assistant workloads. Chroma and Qdrant need a separate process which costs RAM we don't have.

### Why SOPS + age for secrets?

Encrypted secrets committed to the repo mean no `.env` management across devices — clone the repo, provide the age key, and everything is available. Safe for public GitHub since all secrets are encrypted at rest.

### Why `@functools.lru_cache(maxsize=1)` for DB singletons?

Consistent with the `get_settings()` pattern already in the codebase. Lazy initialization (first call, not at import time) means tests can mock `_get_memory` and `_get_skill_lib` cleanly. Module-level instantiation would run DB connections at import time, breaking test isolation.

---

## 15. Testing Strategy

**171 tests, 0 failures** (as of last audit pass).

### Philosophy

- Mock external I/O only: HTTP calls, LLM APIs, Telegram API, file system writes
- Never mock internal functions or domain logic
- Every happy-path test has at least one failure/edge-case companion
- `caplog` for log assertions; `pytest.raises` for exception paths

### Key test files

| File | Tests |
|------|-------|
| `packages/llm/tests/test_gateway.py` | Provider failover, timeout injection, structured output validation |
| `agents/orchestrator/tests/test_graph.py` | HITL flow, routing, sanitize_input integration |
| `agents/spiritual/tests/test_squad.py` | BibleRAG injection defense, verse lookup, reference allowlist |
| `agents/intelligence/tests/test_earnings.py` | Layer 1/2 earnings, FMP fallback, format one-liner |
| `packages/observability/tests/test_tracing.py` | Correlation ID propagation, nested `@traced` non-overwrite |
| `packages/core/tests/test_config.py` | Settings validation, `active_llm_providers()` priority order |

### Running tests

```bash
uv run pytest                         # all tests
uv run pytest packages/llm/           # single package
uv run pytest -k "test_hitl"          # by name filter
uv run pytest --tb=short -q           # compact output
```

### Lint and type check

```bash
ruff check . --fix && ruff format .   # lint + format (zero errors in src/)
ty check .                            # type checker (Astral)
```
