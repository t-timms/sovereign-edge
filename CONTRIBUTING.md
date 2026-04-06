# Contributing to Sovereign Edge

Sovereign Edge is a personal AI system built for single-user deployment. Contributions are welcome for bug fixes, new expert agents, infrastructure improvements, and documentation. This document covers everything needed to open a quality pull request.

---

## Table of Contents

1. [Before You Start](#1-before-you-start)
2. [Development Setup](#2-development-setup)
3. [Branch Strategy](#3-branch-strategy)
4. [The Ship Workflow](#4-the-ship-workflow)
5. [Commit Standards](#5-commit-standards)
6. [Pull Request Checklist](#6-pull-request-checklist)
7. [Code Standards](#7-code-standards)
8. [Adding a New Expert](#8-adding-a-new-expert)
9. [Testing Requirements](#9-testing-requirements)

---

## 1. Before You Start

- **Search existing issues and PRs** before opening a new one.
- **Open an issue first** for significant changes (new experts, architectural shifts, breaking configuration changes). This avoids work in a direction that won't be merged.
- **Small PRs merge faster.** One logical change per PR. Bundle refactors only when splitting creates more churn than value.

---

## 2. Development Setup

```bash
git clone https://github.com/omnipotence-eth/sovereign-edge.git
cd sovereign-edge

# Install all workspace packages in editable mode
uv sync --all-packages

# Pull local models required for intent routing
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b

# Copy and fill in minimum env vars
cp .env.example .env
```

Minimum `.env` for local development:

```bash
SE_TELEGRAM_BOT_TOKEN=<your_bot_token>
SE_TELEGRAM_OWNER_CHAT_ID=<your_numeric_chat_id>
SE_GROQ_API_KEY=<gsk_...>
SE_OLLAMA_HOST=http://localhost:11434
```

---

## 3. Branch Strategy

| Prefix | Use |
|--------|-----|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `refactor/` | Code changes with no behavior change |
| `docs/` | Documentation only |
| `chore/` | Dependency updates, CI, tooling |
| `experiment/` | Exploratory work not ready for main |

Branch from `master`. Open PRs against `master`.

```bash
git checkout -b feature/my-new-expert
```

---

## 4. The Ship Workflow

Every PR must pass through these phases **in order** before committing. This is enforced by pre-commit hooks in the development environment.

### Phase 1 — Lint and format

```bash
ruff check . --fix && ruff format .
```

Zero errors required. Ruff auto-fixes many issues. Remaining errors must be resolved manually or suppressed with a `# noqa: XXXX` comment that includes a justification comment on the same line.

### Phase 2 — Type check

```bash
ty check .
# fallback if ty unavailable:
mypy . --ignore-missing-imports
```

All public function signatures must carry type hints. Use `from __future__ import annotations` at the top of every module.

### Phase 3 — Audit checklist

Before writing the commit message, verify:

- [ ] Type hints on all public signatures
- [ ] Stdlib imports at module level (not inside functions, except for optional deps)
- [ ] No hardcoded secrets — use `pydantic-settings` and env vars
- [ ] No `print()` in production code — use `get_logger(__name__)`
- [ ] Log calls use `logger.info("msg %s", value)` — never f-strings
- [ ] Specific exception types caught — no bare `except:` or `except Exception:` at non-boundary scope
- [ ] External input validated at system boundaries
- [ ] No new SQL string interpolation — parameterized queries only
- [ ] Function length reasonable — extract helpers if a function exceeds ~50 lines

### Phase 4 — Version bump

If you changed any package under `packages/` or `agents/` or `services/`, bump the `version` field in that package's `pyproject.toml`:

- `fix:` → patch bump (`0.3.1` → `0.3.2`)
- `feat:` → minor bump (`0.3.1` → `0.4.0`)
- `feat!:` breaking → major bump (`0.3.1` → `1.0.0`)

### Phase 5 — Commit

Follow [Conventional Commits](#5-commit-standards). Include an `AUDIT:` block in the commit body listing the results of each phase above.

---

## 5. Commit Standards

Format: `<type>(<optional scope>): <short description>`

| Type | Use |
|------|-----|
| `feat` | New capability visible to users |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change with no behavior change |
| `test` | New or updated tests |
| `chore` | Dependency update, CI, tooling, version bump |
| `perf` | Performance improvement |

**Subject line:** ≤72 chars, imperative mood, no trailing period.

**Body:** Required for non-trivial changes. Explain *why*, not *what* — the diff shows what. Include an `AUDIT:` block.

### Example

```
fix: career Jina 422 on ATS-targeted queries

Jina's search API is a semantic endpoint — it does not support Google-style
site: operators or -word exclusions. Queries containing these returned 422
on every career search, leaving the strategist with no live data.

Replaced both ad-hoc and morning-brief query builders with plain
natural-language strings that Jina accepts.

AUDIT:
- ruff check: pass
- ruff format: 1 file reformatted
- ty: pass
- no new public signatures
- no hardcoded secrets
- version bumped: sovereign-edge-career 0.3.1 → 0.3.2

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

---

## 6. Pull Request Checklist

Before marking a PR ready for review:

- [ ] All ship workflow phases completed (lint, type check, audit, version bump)
- [ ] Tests added or updated for the change
- [ ] `pytest -q` passes with zero failures
- [ ] Documentation updated if behavior or configuration changed
- [ ] PR title follows conventional commit format
- [ ] PR description explains the *why* and links to relevant issues

---

## 7. Code Standards

### Python

```python
# ✓ Module-level logger
from observability.logging import get_logger
logger = get_logger(__name__, component="my-component")

# ✓ Lazy log evaluation
logger.info("message %s", value)

# ✗ Never f-strings in log calls
logger.info(f"message {value}")

# ✓ Typed public signatures with __future__ annotations
from __future__ import annotations

async def process(self, task: TaskRequest) -> TaskResult:
    ...

# ✓ Specific exceptions
except httpx.HTTPStatusError as exc:
    ...

# ✗ Never bare except
except Exception:
    ...
```

### Security

- Never concatenate user input into SQL, LLM prompts, or shell commands
- Validate external input at system boundaries with Pydantic or explicit guards
- Use `pydantic-settings` for all configuration — no `os.environ.get()` with defaults scattered through code
- New HTTP endpoints require auth (`SE_DASHBOARD_TOKEN` pattern) and content-type enforcement

### Dependencies

- All new dependencies need an upper version bound: `"package>=X.Y,<(X+1).0"`
- Pin versions for security-sensitive packages (see `litellm==1.82.6` — supply chain incident)
- Run `pip-audit` after any `uv add`

---

## 8. Adding a New Expert

1. Scaffold the package:

```bash
uv init agents/my-expert --lib
```

2. Add it to `[tool.uv.workspace]` in the root `pyproject.toml`.

3. Implement the LangGraph subgraph in `agents/my-expert/src/my_expert/subgraph.py`. Wrap the LangGraph import in `try/except ImportError` so the bot works without it:

```python
try:
    from langgraph.graph import END, START, StateGraph
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    StateGraph = START = END = None
```

4. Implement `BaseExpert` from `core.expert`. The three required methods are `process()`, `morning_brief()`, and `health_check()`.

5. Add the new intent to:
   - `packages/core/src/core/types.py` → `IntentClass` and `Intent` enums
   - `packages/router/src/router/classifier.py` → `_KEYWORD_MAP`, `_INTENT_PROTOTYPES`, `_INTENT_TO_CLASS`
   - `scripts/generate-synth-data.py` → training seeds for the ONNX classifier
   - `scripts/train-router.py` → `_LABEL_NAMES`

6. Register in `services/telegram/src/telegram_bot/bot.py` → `_run()`.

See [Development](docs/development.md) for the full walkthrough with code examples.

---

## 9. Testing Requirements

- Tests live in the `tests/` directory of each package/agent
- Every happy-path test needs at least one failure/edge-case companion
- Mock external I/O (HTTP, LLM APIs, filesystem) — never mock internal functions
- `caplog` for log assertions, `pytest.raises` for exception paths
- Use `pytest.mark.asyncio` (or `asyncio_mode = "auto"` in `pyproject.toml`) for async tests

```bash
# Run all tests
pytest -q

# Run a specific package
pytest packages/llm/ -v

# Run by name filter
pytest -k "test_jina" -v
```

Aim to keep test coverage meaningful rather than maximal — cover edge cases and failure paths, not just happy paths.
