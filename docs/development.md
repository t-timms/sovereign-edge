# Development

Local setup, code quality tooling, testing, and contribution guidelines.

---

## Local Setup

### Prerequisites

- Python 3.11+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Ollama running locally (`ollama serve`)

### Install

```bash
git clone https://github.com/your-username/sovereign-edge.git
cd sovereign-edge
uv sync --all-packages
```

### Environment

Copy and fill in the required variables:

```bash
cp .env.example .env   # or create .env manually
```

Minimum for local development:

```bash
SE_TELEGRAM_BOT_TOKEN=your_token
SE_TELEGRAM_OWNER_CHAT_ID=your_chat_id
SE_GROQ_API_KEY=gsk_...          # or any single cloud key
SE_OLLAMA_HOST=http://localhost:11434
```

### Pull Ollama models

```bash
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
```

### Run the bot

```bash
task start
# or directly:
uv run python -m telegram_bot
```

---

## Project Structure

```
sovereign-edge/
├── agents/
│   ├── career/         # Career expert
│   ├── creative/       # Creative expert
│   ├── intelligence/   # Intelligence expert
│   ├── orchestrator/   # Request dispatch, scheduling, trace store
│   └── spiritual/      # Spiritual expert
├── packages/
│   ├── core/           # Config, types, BaseExpert
│   ├── llm/            # LLM gateway (LiteLLM wrapper)
│   ├── memory/         # Conversation history, semantic cache, episodic memory
│   ├── observability/  # Structured logging
│   ├── router/         # Intent classification, PII detection
│   └── search/         # arXiv, HuggingFace, Jina, Bible API clients
├── services/
│   └── telegram/       # python-telegram-bot handler
├── tests/              # pytest test suite
├── docs/               # This documentation
├── systemd/            # Service unit file
├── secrets/            # Encrypted env vars (SOPS)
├── Taskfile.yml        # Dev task runner
└── pyproject.toml      # Workspace root
```

Each package and agent is a separate uv workspace member with its own `pyproject.toml`. They are installed in editable mode by `uv sync --all-packages`.

---

## Code Quality

### Lint and format

```bash
task lint
# equivalent to:
ruff check . --fix && ruff format .
```

Ruff is configured at the workspace root (`pyproject.toml`):

- Line length: 100
- Rules: E, F, W, I (isort), S (bandit security), B (bugbear), UP (pyupgrade), ANN (type hints), RUF
- Target: Python 3.11

### Type checking

```bash
task typecheck
# equivalent to:
ty check .
```

`ty` (Astral) is the primary type checker — faster than mypy. Fall back to `mypy` if `ty` is unavailable.

All public function signatures require type hints. Use `from __future__ import annotations` at the top of every module.

### Security audit

```bash
task audit
# equivalent to:
pip-audit
```

Run after any dependency change. The Ruff `S` (bandit) ruleset also catches common security issues at lint time.

---

## Testing

```bash
task test
# equivalent to:
pytest -q
```

Tests live in `tests/` and are named `test_*.py`. The pytest configuration (`pyproject.toml`) sets `asyncio_mode = "auto"` so async test functions work without decorators.

### Running a specific test

```bash
pytest tests/test_gateway.py -v
pytest tests/test_router.py::test_pii_detection -v
```

### Test conventions

- Mock external I/O (HTTP, LLM APIs, filesystem) — not internal functions.
- Every happy-path test has at least one failure/edge-case counterpart.
- Use `caplog` to assert log output, not `capsys`.
- The LLM gateway tests use `unittest.mock.patch("litellm.acompletion")` to avoid real API calls.
- LangGraph subgraph tests should cover both the full pipeline path (`subgraph is not None`) and the graceful fallback path (`subgraph is None`). Patch individual nodes to avoid live HTTP calls while still exercising the StateGraph wiring.

---

## Task Reference

All common operations are in `Taskfile.yml`. Run `task --list` to see everything.

| Task | Description |
|---|---|
| `task install` | `uv sync --all-packages` |
| `task lint` | Ruff check + format |
| `task typecheck` | `ty check .` |
| `task test` | `pytest -q` |
| `task audit` | `pip-audit` CVE scan |
| `task start` | Start the bot (foreground) |
| `task stop` | Stop the systemd service |
| `task restart` | Restart the systemd service |
| `task status` | Show service status |
| `task logs` | Follow service logs |
| `task deploy` | rsync to Jetson + restart service |
| `task pull-models` | Pull required Ollama models |

---

## Adding a New Expert

1. Create a new package under `agents/`:

```bash
uv init agents/my-expert --lib
```

2. Add it to the workspace in the root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["agents/my-expert", ...]
```

3. Implement the LangGraph subgraph in `agents/my-expert/src/my_expert/subgraph.py`:

```python
from __future__ import annotations

try:
    from langgraph.graph import END, START, StateGraph
    _LANGGRAPH_AVAILABLE = True
except ImportError:  # graceful degradation — bot still functions without langgraph
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START = END = None

from typing import TypedDict

class MyExpertState(TypedDict):
    query: str
    context: str
    response: str

async def _fetch_node(state: MyExpertState) -> dict:
    ...  # retrieve live data

async def _synthesizer_node(state: MyExpertState) -> dict:
    ...  # call LLM gateway

if _LANGGRAPH_AVAILABLE:
    _graph = StateGraph(MyExpertState)
    _graph.add_node("fetcher", _fetch_node)
    _graph.add_node("synthesizer", _synthesizer_node)
    _graph.add_edge(START, "fetcher")
    _graph.add_edge("fetcher", "synthesizer")
    _graph.add_edge("synthesizer", END)
    my_expert_subgraph = _graph.compile(name="my_expert")
else:
    my_expert_subgraph = None
```

Follow the pattern used by all existing experts: wrap the import in `try/except ImportError`, export `None` when unavailable, and check before invoking in `expert.process()`.

4. Implement `BaseExpert` from `core.expert`, delegating to the subgraph:

```python
from core.expert import BaseExpert
from core.types import ExpertName, TaskRequest, TaskResult
from .subgraph import my_expert_subgraph

class MyExpert(BaseExpert):
    @property
    def name(self) -> str:
        return ExpertName.MY_EXPERT  # add to ExpertName enum in core/types.py

    async def process(self, task: TaskRequest) -> TaskResult:
        if my_expert_subgraph is not None:
            state = await my_expert_subgraph.ainvoke({"query": task.content, ...})
            return TaskResult(content=state["response"])
        # fallback: direct LLM call when LangGraph is unavailable
        ...

    async def morning_brief(self) -> str:
        ...

    async def health_check(self) -> bool:
        ...
```

5. Register the expert in `agents/orchestrator/src/orchestrator/main.py` and add the intent mapping in `packages/router/src/router/classifier.py`.

---

## Structured Logging

All components use `structlog` via the shared logger:

```python
from observability.logging import get_logger

logger = get_logger(__name__, component="my-component")

logger.info("event_name", key="value", count=42)
```

In production (systemd), output is JSON. In development, it is colorized plain text. Set `LOG_JSON=true` to force JSON locally.

Every log line automatically carries `component`, `expert`, and `model` context fields for filtering in `journalctl`:

```bash
journalctl -u telegram-bot | grep '"component":"career"'
```

---

## Troubleshooting

Common deployment errors and their fixes are documented in [Troubleshooting](troubleshooting.md).

---

## Conventional Commits

```
feat: add episodic memory search to career expert
fix: handle empty arXiv response gracefully
docs: add deployment guide
chore: bump LiteLLM to 1.82.7
refactor: extract rate-limit check into gateway method
test: add fallback chain integration test
```

Branch naming: `feature/`, `fix/`, `experiment/`, `chore/`
