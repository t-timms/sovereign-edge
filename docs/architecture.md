# Architecture

## Overview

Sovereign Edge is a monorepo of Python packages wired together into a single async service. The Telegram and Discord bots are the entry points; either can be used independently. Requests flow through an intent router, are dispatched to an expert, grounded with live data, processed through the LLM gateway, and returned — with every call traced to SQLite.

The system is designed around two principles: **graceful degradation** (every layer has a fallback) and **privacy by default** (PII is detected before routing and kept local).

---

## Request Flow

```
User message (Telegram)
    │
    ├─ Input validation: length cap (2000 chars), rate limit (2s/chat)
    │
    ▼
PIIDetector.contains_pii()
    ├─ PII found → RoutingDecision.LOCAL (Ollama only, no external calls)
    └─ No PII → RoutingDecision.CLOUD
    │
    ▼
IntentRouter.aroute()  [3-tier classification]
    1. Embedding cosine similarity (Ollama embeddings vs. prototype sentences)
       └─ score ≥ 0.45 → use this result
    2. ONNX DistilBERT classifier (if model loaded)
    3. Keyword matching (weighted, multi-word keys first)
    │
    ▼
Orchestrator.dispatch()
    ├─ Inject conversation history (last 8 turns from SQLite)
    ├─ Semantic cache lookup (cosine similarity ≥ 0.92 → return cached)
    └─ Route to expert by intent
    │
    ▼
Expert.process()
    ├─ Fetch live data in parallel (arXiv + HF, or Jina, or Bible API)
    └─ Build messages with system prompt + history + live context + user input
    │
    ▼
LLMGateway.complete()
    Groq → Gemini → Cerebras → Mistral → Ollama (local fallback)
    Each provider: token-bucket RPM check → daily token cap check → call with retry
    │
    ▼
Orchestrator (post-dispatch)
    ├─ Store response in semantic cache
    ├─ Append turn to conversation history
    └─ Record trace (model, expert, tokens, latency, cost)
    │
    ▼
Telegram reply (chunked at 4000 chars)
```

---

## LLM Gateway

The gateway (`packages/llm`) wraps LiteLLM as a library (not a proxy) and manages a priority-ordered provider chain.

### Fallback Chain

| Priority | Provider | Model | RPM | Daily Tokens |
|---|---|---|---|---|
| 1 | Groq | `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | 500K |
| 2 | Gemini | `gemini-2.5-flash` | 15 | 250K |
| 3 | Cerebras | `llama-3.3-70b` | 30 | 1M |
| 4 | Mistral | `mistral-small-latest` | 2 | 33M |
| 5 | Ollama | `qwen3:0.6b` (local) | unlimited | unlimited |

### Rate Limiting

Each provider has an independent token bucket. The bucket refills at `rpm / 60` tokens per second and holds a maximum of `rpm` tokens (one minute burst). If a provider's bucket is empty or its daily token cap is reached, the gateway skips it and tries the next.

### Error Handling

| Error | Behavior |
|---|---|
| `RateLimitError` | Exponential backoff (1s, 2s, 4s), then skip provider |
| `ServiceUnavailableError` | Exponential backoff, then skip provider |
| `AuthenticationError` | Skip provider immediately (bad API key) |
| `BadRequestError` | Skip provider immediately |
| `TimeoutError` | Exponential backoff, then skip provider |
| All providers fail | Fall back to local Ollama |

The gateway is a module-level singleton (`get_gateway()`). Direct instantiation bypasses rate limiting state.

---

## Intent Router

The router (`packages/router`) classifies each message into one of five intents: `SPIRITUAL`, `CAREER`, `INTELLIGENCE`, `CREATIVE`, `GENERAL`.

### Three-Tier Classification

**Tier 1 — Embedding similarity (async):**
The query is embedded with Ollama (`qwen3-embedding:0.6b`) and scored against prototype sentences for each intent via cosine similarity. If the best score exceeds 0.45, this result is used. Falls back to Tier 2 on Ollama failure.

**Tier 2 — ONNX DistilBERT (sync):**
A fine-tuned DistilBERT model served via ONNX runtime. Loaded at startup from `data/models/router.onnx`. If the model file is absent, skips to Tier 3.

**Tier 3 — Keyword matching (sync):**
Weighted keyword matching. Multi-word keys are checked before single-word keys. Confidence is derived from match weight. If no keywords match, returns `Intent.GENERAL` at 0.5 confidence.

### PII Routing

PII detection runs before classification. If SSN, credit card, email, phone, or IP address patterns are detected, the routing decision is forced to `LOCAL` regardless of intent. No external API calls are made.

---

## Memory Layers

### Conversation History
SQLite database (WAL mode). Stores up to 40 turns per `chat_id`. The 8 most recent turns are injected into every request as prior message context, giving experts short-term conversational memory.

### Semantic Cache
LanceDB vector store. After every cloud LLM call, the query and response are stored with their embedding vector. On subsequent requests, the query is embedded and searched for cosine similarity ≥ 0.92. A cache hit returns the stored response without calling the LLM. Cache entries expire after 24 hours.

### Episodic Memory (optional)
Long-term episodic memory via Mem0. Requires the `mem0ai` optional dependency. Extracts and stores facts from conversations, searchable by semantic similarity. Silently disabled when the dependency is unavailable.

---

## Morning Pipeline

The orchestrator uses APScheduler to fire six cron jobs per day (Central Time):

```
05:00  _morning_health_check    All experts pinged in parallel
05:15  _spiritual_brief         Live Bible verse → morning devotional
05:30  _intelligence_brief      arXiv + HuggingFace papers → digest
06:00  _career_brief            Job market search → actionable brief
07:00  _creative_brief          Trend context → daily creative prompt
18:00  _career_rescan_brief     Evening job scan
```

Each step calls `expert.morning_brief()` with a 90-second timeout. Output is chunked at 4000 characters and pushed to the owner's Telegram chat via `send_message()`.

---

## Observability

Every completed task is recorded to a SQLite trace store (WAL mode) with:

- `task_id`, `timestamp`, `expert`, `model`
- `tokens_in`, `tokens_out`, `latency_ms`, `cost_usd`
- `cached` (bool), `routing` (LOCAL/CLOUD/CACHE), `status`, `error_message`

The `/stats` Telegram command queries today's aggregated totals: requests, cache hits, errors, average latency, total tokens, total cost, and models used.

Structured logging uses structlog with JSON output in production. Every log line carries `component`, `expert`, and `model` context fields for filtering.

---

## Security

- **Auth:** All Telegram commands and messages validate `chat_id` against `SE_TELEGRAM_OWNER_CHAT_ID`. Unauthorized attempts are logged at CRITICAL.
- **Rate limiting:** Per-chat 2-second cooldown prevents request flooding.
- **Input cap:** Messages truncated at 2000 characters before processing.
- **PII guard:** PII detected → forced local routing, no external API calls.
- **SSRF guard:** Jina `fetch()` rejects URLs resolving to RFC 1918 / loopback addresses (fail-closed on DNS error).
- **Prompt injection:** User input is wrapped in `<user_request>` XML delimiters in all expert prompts.
- **Secrets:** SOPS Age encryption. Decrypted to a `600`-permission file at service startup only.
- **Supply chain:** HuggingFace model pinned to a specific commit hash. LiteLLM pinned to 1.82.6.
