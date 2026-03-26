# Sovereign Edge

A personal AI intelligence system — privacy-first, edge-deployed, and grounded in live data.

Sovereign Edge runs four specialized AI agents on a Jetson Nano, accessible through Telegram. Every response is grounded with real-time data: live arXiv papers, HuggingFace Daily Papers, live Bible verse retrieval, and Jina web search. PII stays local. Cloud APIs are free-tier only. A scheduled morning pipeline delivers actionable briefs before the workday starts.

---

## Squads

| Squad | Purpose | Live Data Source |
|---|---|---|
| **Intelligence** | AI/ML research synthesis, trend monitoring | arXiv, HuggingFace Daily Papers |
| **Career** | Job search, resume coaching, interview prep | Jina web search |
| **Creative** | Writing, content strategy, social media | Jina web search |
| **Spiritual** | Scripture study, prayer, devotionals | bible-api.com (KJV) |

---

## Morning Pipeline

Delivered automatically to Telegram each day (Central Time):

| Time | Brief |
|---|---|
| 05:00 | Health check — all squads validated |
| 05:15 | Morning devotional with live scripture |
| 05:30 | AI/ML intelligence digest (arXiv + HuggingFace) |
| 06:00 | Career brief — DFW job market scan |
| 07:00 | Creative direction — daily content prompt |
| 18:00 | Evening career rescan |

---

## Architecture

```
Telegram (owner-only)
        │
        ▼
  ┌─────────────────────────────────────────┐
  │           Intent Router                 │
  │  1. Embedding similarity (Ollama)       │
  │  2. ONNX DistilBERT classifier          │
  │  3. Keyword fallback                    │
  │  4. PII check → force LOCAL             │
  └──────────────┬──────────────────────────┘
                 │
        ┌────────▼────────┐
        │   Orchestrator  │  ← APScheduler (morning pipeline)
        │   + TraceStore  │  ← SQLite (cost, latency, tokens)
        └────────┬────────┘
                 │
    ┌────────────┼────────────┬─────────────┐
    ▼            ▼            ▼             ▼
 Spiritual    Career    Intelligence    Creative
  Squad        Squad       Squad          Squad
    │            │            │             │
    ▼            ▼            ▼             ▼
Bible API    Jina Search   arXiv +       Jina Search
  (KJV)                   HF Papers
                 │
        ┌────────▼────────────────────────────┐
        │           LLM Gateway               │
        │  Groq → Gemini → Cerebras →         │
        │  Mistral → Ollama (local fallback)  │
        └─────────────────────────────────────┘
                 │
        ┌────────▼────────────────────────────┐
        │              Memory                 │
        │  Conversation history (SQLite)      │
        │  Semantic cache (LanceDB)           │
        │  Episodic memory (Mem0 / optional)  │
        └─────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.11, asyncio |
| Package management | uv (workspace) |
| LLM routing | LiteLLM 1.82.6 |
| Cloud providers | Groq, Gemini, Cerebras, Mistral (all free tier) |
| Local inference | Ollama (`qwen3:0.6b`) |
| Embeddings | Ollama (`qwen3-embedding:0.6b`) |
| Intent classification | ONNX DistilBERT + keyword fallback |
| Vector store | LanceDB |
| Conversation memory | SQLite (WAL mode) |
| Observability | structlog + SQLite trace store |
| Secrets | SOPS (Age encryption) |
| Scheduling | APScheduler |
| Interface | python-telegram-bot |
| Deployment | systemd on Jetson Nano |

---

## Quick Start

```bash
git clone https://github.com/omnipotence-eth/sovereign-edge.git
cd sovereign-edge

# Install dependencies
uv sync --all-packages

# Pull local models
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b

# Configure environment
cp .env.example .env        # fill in API keys and Telegram credentials

# Personalize your instance (optional but recommended)
cp SOUL.md.example SOUL.md  # define your mission, squads, and values

# Run
uv run python -m telegram_bot
```

See [Configuration](docs/configuration.md) for all `SE_` environment variables and [Deployment](docs/deployment.md) for the full Jetson + systemd setup.

---

## Personalization

Sovereign Edge is designed to serve one person. Two files make it yours:

**`.env`** — API keys, Telegram credentials, storage paths. Copy from `.env.example`.

**`SOUL.md`** — Your identity document: who the system serves, what each squad is optimized for, and what values govern it. Copy from `SOUL.md.example`. This file is gitignored and never leaves your machine.

The career squad system prompt in `agents/career/src/career/squad.py` is the other place to personalize — update it with your target roles, market, and differentiators.

---

## Documentation

- [Architecture](docs/architecture.md) — request flow, memory layers, LLM gateway
- [Squads](docs/squads.md) — each agent's capabilities and data sources
- [Configuration](docs/configuration.md) — all environment variables
- [Deployment](docs/deployment.md) — Jetson setup, systemd, secrets
- [Development](docs/development.md) — local setup, testing, code quality

---

## License

MIT
