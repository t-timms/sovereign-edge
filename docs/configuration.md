# Configuration

All configuration is loaded from environment variables via Pydantic `BaseSettings`. The prefix for every variable is `SE_`.

On a local dev machine, place variables in a `.env` file at the project root. On the Jetson, the SOPS-encrypted `secrets/env.yaml` is decrypted at service startup and written to `secrets/.env.decrypted`.

---

## Required Variables

The service will not function correctly without these.

| Variable | Description |
|---|---|
| `SE_TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `SE_TELEGRAM_OWNER_CHAT_ID` | Your numeric Telegram chat ID — only this ID can interact with the bot |

At least one cloud LLM key is required for cloud routing. All four enables the full fallback chain.

| Variable | Provider | Free Tier |
|---|---|---|
| `SE_GROQ_API_KEY` | Groq (Priority 1) | 500K tokens/day, 30 RPM |
| `SE_GOOGLE_API_KEY` | Gemini (Priority 2) | 250K tokens/day, 15 RPM |
| `SE_CEREBRAS_API_KEY` | Cerebras (Priority 3) | 1M tokens/day, 30 RPM |
| `SE_MISTRAL_API_KEY` | Mistral (Priority 4) | 33M tokens/day, 2 RPM |

---

## Discord (Optional)

To enable the Discord bot in addition to (or instead of) Telegram, set:

| Variable | Description |
|---|---|
| `SE_DISCORD_BOT_TOKEN` | Bot token from Discord Developer Portal |
| `SE_DISCORD_OWNER_USER_ID` | Your numeric Discord user ID — only this user can interact with the bot |

---

## Optional API Keys

| Variable | Description |
|---|---|
| `SE_JINA_API_KEY` | Jina AI Reader — removes the ~200 RPD cap on web search; free plan available at jina.ai |
| `SE_ALPHA_VANTAGE_KEY` | Alpha Vantage — financial data for career and intelligence agents |

---

## Ollama (Local Inference)

| Variable | Default | Description |
|---|---|---|
| `SE_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API endpoint (use `http://ollama:11434` in Docker Compose) |
| `SE_LOCAL_LLM_MODEL` | `qwen3:0.6b` | Model for chat completions (local fallback) |
| `SE_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Model for intent routing embeddings |

Ollama is the Priority 5 fallback — it is always available and has no rate limit. It is also the only provider used when routing is forced `LOCAL` (PII detected).

---

## Storage Paths

| Variable | Default | Description |
|---|---|---|
| `SE_PROJECT_ROOT` | `~/sovereign-edge` | Root of the sovereign-edge repo |
| `SE_SSD_ROOT` | `~/sovereign-edge/data` | Base directory for persistent data |
| `SE_LANCEDB_PATH` | `{ssd_root}/lancedb` | Semantic cache vector store |
| `SE_LOGS_PATH` | `{ssd_root}/logs` | Structured log output |
| `SE_MODELS_PATH` | `{ssd_root}/models` | ONNX router model and other artifacts |

On Jetson, set `SE_SSD_ROOT` to a path on the NVMe SSD (e.g., `/mnt/ssd/sovereign-edge-data`) to avoid wearing out the eMMC.

---

## Rate Limits

These control the per-provider token bucket. Override only if your actual API tier differs from the defaults.

| Variable | Default | Description |
|---|---|---|
| `SE_GROQ_RPM` | `30` | Groq requests per minute |
| `SE_GEMINI_RPM` | `15` | Gemini RPM |
| `SE_CEREBRAS_RPM` | `30` | Cerebras RPM |
| `SE_MISTRAL_RPM` | `2` | Mistral RPM |

Daily token caps per provider are set in `packages/llm/src/llm/gateway.py` (`tpd` field on each `ProviderConfig`). They are not overridable via environment variables.

---

## Career Personalization

| Variable | Default | Description |
|---|---|---|
| `SE_CAREER_TARGET_LOCATION` | `your city or region` | Geographic focus for job searches |
| `SE_CAREER_TARGET_ROLES` | `ML Engineer, AI Engineer, LLM Engineer` | Comma-separated target job titles |
| `SE_CAREER_DIFFERENTIATORS` | *(empty)* | Comma-separated skills/differentiators to highlight in coaching |

---

## Feature Flags

| Variable | Default | Description |
|---|---|---|
| `SE_VOICE_ENABLED` | `false` | Enable voice message processing (requires additional dependencies) |
| `SE_CREATIVE_ENABLED` | `true` | Enable the creative expert |
| `SE_DEBUG_MODE` | `false` | Verbose debug logging — enable only during development |

---

## Scheduling

| Variable | Default | Description |
|---|---|---|
| `SE_MORNING_WAKE_HOUR` | `5` | Hour (0–23) for the morning pipeline start |
| `SE_MORNING_WAKE_MINUTE` | `0` | Minute (0–59) for the morning pipeline start |
| `SE_TIMEZONE` | `US/Central` | pytz timezone string used by APScheduler |

The full morning pipeline schedule fires at offsets from `SE_MORNING_WAKE_HOUR`. Changing this variable shifts the entire pipeline.

---

## Observability

| Variable | Default | Description |
|---|---|---|
| `LOG_JSON` | `false` | Emit logs as JSON (recommended in production for log aggregation) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(empty)* | OpenTelemetry collector endpoint — leave blank to disable tracing export |

---

## Example `.env`

```bash
# Telegram
SE_TELEGRAM_BOT_TOKEN=your_bot_token_here
SE_TELEGRAM_OWNER_CHAT_ID=your_chat_id_here

# Cloud LLM keys (all free tier)
SE_GROQ_API_KEY=gsk_...
SE_GOOGLE_API_KEY=AIza...
SE_CEREBRAS_API_KEY=csk_...
SE_MISTRAL_API_KEY=...

# Optional search key
SE_JINA_API_KEY=jina_...

# Storage (Jetson — use SSD path)
SE_SSD_ROOT=/mnt/ssd/sovereign-edge-data

# Ollama
SE_OLLAMA_HOST=http://localhost:11434

# Career targeting
SE_CAREER_TARGET_LOCATION=Austin, TX
SE_CAREER_TARGET_ROLES=ML Engineer, AI Engineer, LLM Engineer
SE_CAREER_DIFFERENTIATORS=GRPO fine-tuning, LangGraph agents, vLLM serving

# Scheduling
SE_TIMEZONE=US/Central
SE_MORNING_WAKE_HOUR=5
```
