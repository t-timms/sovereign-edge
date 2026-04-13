# Sovereign Edge — Claude Code Project Guide

This file gives Claude Code complete context about the sovereign-edge system.
Read this before making any changes. It is the authoritative reference for architecture,
conventions, running, testing, and deployment.

---

## What This Is

A single-user, privacy-first personal AI system deployed on a **Jetson Orin Nano** (ARM64,
JetPack 6). Five specialized expert agents handle distinct life domains. All requests flow
through a 3-tier intent router, a LangGraph director, and a multi-provider LLM gateway with
automatic failover. Morning briefs are delivered automatically before the workday starts.

Owner: `Tremayne Timms` — one Telegram chat ID, one bot, always-on.

---

## Architecture at a Glance

```
Telegram message
    → auth (owner-only) + rate limit (2s gap) + 2000-char cap
    → 3-tier router: Ollama embeddings → ONNX DistilBERT → keyword fallback
    → PII check → force LOCAL routing if detected
    → LangGraph director (cross-expert query planning)
    → Expert dispatch (one or more subgraphs in parallel)
    → LLM gateway: Groq → Gemini → Mistral → Ollama
    → Memory: episodic (Mem0) + semantic cache (LanceDB) + SQLite
    → Telegram response
```

---

## Repo Layout

```
sovereign-edge/
├── agents/
│   ├── career/         # Jina web search → job listings → structured JobListingResponse
│   ├── creative/       # Jina web search → content ideas → synthesis
│   ├── goals/          # SQLite goal store → top 3 + action
│   ├── intelligence/   # arXiv + HuggingFace papers → flashrank reranker → IntelBriefResponse
│   ├── orchestrator/   # Director LangGraph + APScheduler (7 cron jobs) + trace store
│   └── spiritual/      # bible-api.com → scripture + devotional
├── packages/
│   ├── core/           # Settings (pydantic-settings), types (ExpertName, RoutingDecision), BaseExpert
│   ├── llm/            # LLMGateway: LiteLLM wrapper, token buckets, UsageTracker, structured output
│   ├── memory/         # ConversationStore (SQLite), SemanticCache (LanceDB), EpisodicMemory (Mem0)
│   ├── observability/  # structlog setup, SQLite trace store
│   ├── router/         # IntentClassifier (3-tier), PIIDetector
│   └── search/         # arXiv, HuggingFace, Jina, bible-api.com clients
├── services/
│   ├── discord/        # discord.py handler (owner whitelist)
│   ├── health/         # /health + /ready HTTP endpoints
│   ├── mcp/            # MCP SSE server (ask_expert, get_memory, get_goals tools)
│   ├── telegram/       # python-telegram-bot handler + single-instance flock guard
│   ├── voice/          # Voice message pipeline (optional)
│   └── whatsapp/       # Twilio webhook handler
├── .github/
│   ├── workflows/ci.yml          # lint → test on push/PR
│   ├── workflows/release.yml     # vX.Y.Z tag → GitHub Release
│   ├── ISSUE_TEMPLATE/           # bug_report.yml, feature_request.yml
│   └── PULL_REQUEST_TEMPLATE.md  # ship checklist
├── systemd/
│   └── telegram-bot.service      # systemd unit with placeholder substitution
├── secrets/
│   └── env.yaml                  # SOPS+Age encrypted — NEVER commit plaintext
├── docs/                         # deployment, development, configuration, architecture, troubleshooting
├── ARCHITECTURE.md               # 15-section deep reference
├── CHANGELOG.md                  # Keep a Changelog format
├── CONTRIBUTING.md               # Ship workflow, audit checklist, conventional commits
├── SECURITY.md                   # Security model + incident history
└── pyproject.toml                # uv workspace root
```

---

## Key Files to Know

| File | Why it matters |
|------|---------------|
| `packages/core/src/core/config.py` | All `SE_` env vars with defaults — read this before adding any new config |
| `packages/core/src/core/types.py` | `ExpertName`, `RoutingDecision`, `TaskRequest`, `TaskResult`, `SquadState` |
| `packages/llm/src/llm/gateway.py` | Provider chain, token buckets, `complete()`, `complete_structured()`, `stream_complete()` |
| `packages/router/src/router/classifier.py` | 3-tier intent classification + keyword lists per expert |
| `agents/orchestrator/src/orchestrator/main.py` | Expert registry, `dispatch()`, morning schedule |
| `packages/memory/src/memory/conversation.py` | `ConversationStore` — token budget guard at `MAX_HISTORY_TOKENS` |

---

## Running Locally

```bash
# Prerequisites: Python 3.11+, uv, Ollama running

git clone https://github.com/omnipotence-eth/sovereign-edge.git
cd sovereign-edge
uv sync --all-packages

# Pull local models
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b

# Minimum .env for local dev
SE_TELEGRAM_BOT_TOKEN=your_token
SE_TELEGRAM_OWNER_CHAT_ID=your_chat_id
SE_GROQ_API_KEY=gsk_...
SE_OLLAMA_HOST=http://localhost:11434

# Run
task start
# or: uv run python -m telegram_bot
```

---

## Testing

```bash
task test
# equivalent: uv run pytest -q --tb=short

# With coverage
uv run pytest -q --cov=packages --cov=agents --cov=services --cov-report=term-missing

# Specific test
uv run pytest agents/intelligence/tests/test_arxiv.py -v
```

Known stub tests — do NOT fix unless implementing the feature:
- `agents/creative/tests/test_squad.py`
- `agents/spiritual/tests/test_squad.py`
- `packages/observability/tests/test_tracing.py`

CI ignores these via `--ignore=` flags.

Test conventions:
- Mock all external I/O (HTTP, LLM APIs, filesystem) — not internal functions
- Every happy-path test needs at least one failure/edge-case counterpart
- Use `caplog` for log assertions
- LangGraph subgraph tests must cover both `subgraph is not None` and fallback path

---

## Deployment (Jetson Orin)

```bash
# From dev machine
task deploy
# rsync to Jetson + ssh restart

# On Jetson
cd ~/sovereign-edge && git pull && sudo systemctl restart telegram-bot.service

# Logs
journalctl -u telegram-bot -f
journalctl -u telegram-bot --since today | python3 -m json.tool
```

Secrets are SOPS+Age encrypted in `secrets/env.yaml`. The systemd service decrypts on startup.
The bot has a `fcntl.flock` single-instance guard — starting twice is safe.

---

## Task Commands

| Command | What it does |
|---------|-------------|
| `task install` | `uv sync --all-packages` |
| `task lint` | `ruff check . --fix && ruff format .` |
| `task typecheck` | `ty check .` (falls back to mypy) |
| `task test` | `pytest -q` |
| `task audit` | `pip-audit` CVE scan |
| `task start` | Run bot in foreground |
| `task deploy` | rsync to Jetson + restart service |
| `task pull-models` | Pull required Ollama models |

---

## LLM Provider Chain

| Priority | Provider | Model | Structured? | Free Limit |
|----------|---------|-------|-------------|------------|
| 1 | Groq | llama-4-scout-17b | No | 500K TPD |
| 2 | Gemini | gemini-2.5-flash | Yes (thinking model) | 250K TPD |
| 3 | Mistral | mistral-small-latest | No | 33M TPD |
| 4 | Ollama | qwen3:0.6b | No | Unlimited (local) |

Structured output (instructor + Pydantic) is Gemini-only. All others return unstructured text.
PII detection forces routing to Ollama only — no data leaves the device.

---

## Known Constraints

| Issue | Status |
|-------|--------|
| LiteLLM `1.82.7/1.82.8` supply chain attack | Fixed — upgraded to `>=1.83.0` |
| Cerebras always 404 | Removed from provider chain entirely (2025-12). Re-add when stable. |
| flashrank `ms-marco-MiniLM-L-4-v2` removed from HuggingFace | Fixed — switched to `ms-marco-TinyBERT-L-2-v2` |
| mem0 optional dependency | `uv add "sovereign-edge-memory[episodic]"` to enable — conflicts with openai v2 in older mem0 versions |
| ONNX router model | `data/models/router.onnx` required — falls back to keyword matching if absent |
| `uv.lock` not on Jetson | Run `uv sync --all-packages` after `git pull` when new deps added |

---

## Adding a New Expert

1. `uv init agents/my-expert --lib`
2. Add to `[tool.uv.workspace]` in root `pyproject.toml`
3. Implement `BaseExpert` from `core.expert` + LangGraph subgraph in `subgraph.py`
4. Wrap LangGraph import in `try/except ImportError` — export `None` when unavailable
5. Register in `agents/orchestrator/src/orchestrator/main.py`
6. Add intent mapping in `packages/router/src/router/classifier.py`
7. Add `ExpertName.MY_EXPERT` to `packages/core/src/core/types.py`

See `docs/development.md#adding-a-new-expert` for the full template.

---

## How Claude Code Should Approach Changes

1. **Read before writing** — always read the affected file(s) before editing. Never guess at structure.
2. **Check `core/config.py` first** — before adding env vars, check if one already exists.
3. **Check `core/types.py` first** — before adding types, check `ExpertName`, `RoutingDecision`, `SquadState`.
4. **Bump the package version** — every code change requires a version bump in the affected `pyproject.toml`.
5. **Update CHANGELOG.md** — add the change under `[Unreleased]` before committing.
6. **Run ruff before committing** — `ruff check . --fix && ruff format .`
7. **Follow the ship workflow** in `CONTRIBUTING.md` — ruff → ty → audit → version bump → conventional commit with AUDIT block.
8. **Never touch `secrets/`** — encrypted files only, no plaintext secrets.
9. **Test on both paths** — if a LangGraph subgraph is modified, test both `subgraph is not None` and the fallback.
10. **Deployment is separate from merge** — push to master, then `git pull` on Jetson manually.
