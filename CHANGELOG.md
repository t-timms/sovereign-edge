# Changelog

All notable changes to Sovereign Edge are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `CLAUDE.md` (project root): comprehensive Claude Code guide — architecture, key files, running, testing, deployment, known constraints, and per-change conventions.
- `.pre-commit-config.yaml`: ruff + ruff-format + pre-commit-hooks run automatically on every commit.
- `.github/dependabot.yml`: weekly grouped dependency PRs; LiteLLM pinned exception documented.
- `.github/workflows/release.yml`: tag `vX.Y.Z` → GitHub Release with CHANGELOG entry extracted automatically.
- `MAX_HISTORY_TOKENS` budget in `ConversationStore.get_recent()` — trims oldest turns so history never exceeds ~2000 estimated tokens per request.
- `pytest-cov` + `pre-commit` added as workspace dev dependencies.
- Coverage reporting (`--cov` + `--cov-report=term-missing`) added to CI test job.

### Changed
- All workspace package versions aligned to `0.3.6` (were split across `0.1.0`–`0.3.5`; services/discord, services/mcp, services/whatsapp, agents/goals lagged furthest).

### Fixed
- `packages/search/src/search/jina.py`: replace bare `except Exception` in `_is_private_address` with `(TimeoutError, OSError, ValueError)` — prevents swallowing `CancelledError` in async context.
- `agents/career/src/career/subgraph.py`: replace bare `except Exception` in `_check_url_live` with `httpx.RequestError` — only network errors are expected in this HTTP HEAD check.
- `services/telegram/src/telegram_bot/bot.py`: replace `print()` in single-instance flock guard with structured `logger.error()`.
- Remove four stale `noqa` directives flagged by ruff after prior fixes.

---

### Changed (prior)
- `uv.lock` removed from `.gitignore` — lock file is now committed for reproducible builds.
- `litellm` unpinned from `1.82.6` — upgraded to `>=1.83.0,<2.0.0`, skipping the compromised 1.82.7/1.82.8 range. See `SECURITY.md` for incident details.
- `dependabot.yml` LiteLLM ignore exception removed — future upgrades are welcome.
- `SECURITY.md` updated with the March 2026 supply chain incident (`.claude/settings.json` injection attack).

---

## [0.3.4] — 2026-04-06

### Fixed
- **Intelligence ranker**: `ms-marco-MiniLM-L-4-v2.zip` was removed from the `prithivida/flashrank` HuggingFace repo, causing a 404 on first load and silent fallback to keyword ranking. Replaced with `ms-marco-TinyBERT-L-2-v2` (same ~4 MB ONNX CPU model, ARM64-compatible).

---

## [0.3.3] — 2026-04-05

### Fixed
- **Mistral structured output**: Mistral API returns free-text markdown instead of tool calls for complex structured requests. `instructor`'s `parse_tools` asserted `len(tool_calls) == 1`, which raised `AssertionError` when `tool_calls=None`. Marked `supports_structured=False` on `ProviderConfig` — Mistral is now skipped in the structured chain entirely. Structured output is Gemini-only.
- **Mistral LLM gateway docstring**: Updated provider priority documentation to reflect that Mistral is unstructured-only.

---

## [0.3.2] — 2026-04-05

### Fixed
- **Career Jina 422**: Career subgraph used Google-style `site:greenhouse.io OR ...` and `-word` search operators. Jina's semantic search API rejects these with 422 Unprocessable Entity. Replaced all career search queries with plain natural-language strings.
- **Bible bare-book 404**: `extract_reference()` regex matched bare book abbreviations (e.g. `"prov"`) with no chapter number. `bible-api.com` returns 404 for bare book names. Function now returns `None` when the matched text contains no digits, falling through to `random_verse()`.
- **Duplicate bot instances**: Added `fcntl.flock(LOCK_EX | LOCK_NB)` single-instance guard to `bot.main()`. OS releases the lock on process exit/crash — no stale lock files.

### Added
- **`.github/` templates**: CI workflow, bug report and feature request issue templates, PR template.
- **`CHANGELOG.md`**: This file.
- **`SECURITY.md`**: Vulnerability disclosure policy.

---

## [0.3.1] — 2026-04-04

### Added
- **Goals expert**: New `agents/goals` agent — SQLite-backed personal goal store with `add_goal`, `list_goals`, `update_goal`, `morning_brief` (top 3 urgent goals + one concrete action each). Registered in orchestrator and router.
- **Director graph**: Cross-expert query planning in `agents/orchestrator` — LangGraph `StateGraph` that decomposes multi-domain queries and dispatches to multiple experts in parallel.
- **MCP server**: `services/mcp` — SSE transport, `ask_expert`, `get_memory`, `get_goals` tools. `SE_MCP_ENABLED=true` to activate.
- **WhatsApp service**: `services/whatsapp` — Twilio webhook, owner whitelist, same expert routing as Telegram.
- **Health service**: `services/health` — `/health` and `/ready` HTTP endpoints for external monitoring.
- **Web dashboard**: `/api/v1/traces`, `/api/v1/stats` — read-only, bearer-token protected. `SE_DASHBOARD_TOKEN` required.
- **ONNX router**: DistilBERT INT8 cross-encoder for 6-class intent classification (<10ms on Jetson CPU). `data/models/router.onnx` required; falls back to keyword matching if absent.
- **Episodic memory**: Mem0 integration in `packages/memory` — semantic recall across sessions (optional dependency).

### Changed
- **Full documentation overhaul**: New `README.md` (Mermaid diagram, badges, collapsible stack), `CONTRIBUTING.md` (ship workflow, audit checklist), `docs/troubleshooting.md` (real deployment errors), `docs/architecture.md` (Mermaid quick-reference).
- **Repo cleanup**: Author metadata filled in, all placeholder strings replaced with `<angle_bracket>` convention, `.gitignore` updated with tooling caches.

---

## [0.3.0] — 2026-03-01

### Added
- Initial LangGraph multi-agent architecture — four expert subgraphs (Spiritual, Career, Intelligence, Creative), each with a multi-node fetch → synthesize pipeline.
- LiteLLM gateway with four cloud providers (Groq, Gemini, Cerebras, Mistral) and Ollama local fallback, token-bucket rate limiting, and automatic failover.
- Three-tier intent router: embedding similarity (Tier 1) → ONNX DistilBERT (Tier 2) → keyword fallback (Tier 3). PII detection forces local-only routing.
- Semantic cache via LanceDB — embedding similarity lookup before LLM call.
- APScheduler morning pipeline — 7 cron jobs from 05:00–18:00 CT.
- SOPS + Age secrets management — encrypted `secrets/env.yaml` decrypted at service startup.
- systemd service with `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`, memory and CPU limits.
- Structured output via `instructor` + Pydantic — `JobListingResponse`, `IntelBriefResponse`.

[Unreleased]: https://github.com/omnipotence-eth/sovereign-edge/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/omnipotence-eth/sovereign-edge/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/omnipotence-eth/sovereign-edge/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/omnipotence-eth/sovereign-edge/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/omnipotence-eth/sovereign-edge/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/omnipotence-eth/sovereign-edge/releases/tag/v0.3.0
