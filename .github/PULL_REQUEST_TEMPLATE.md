## What changed

<!-- Bullet list of what this PR does -->

## Why

<!-- Motivation — the problem this solves or the feature it adds -->

## Ship checklist

- [ ] `ruff check . --fix && ruff format .` — clean
- [ ] `ty check .` (or `mypy` fallback) — no new errors
- [ ] Full audit: type hints on public APIs, no bare `except`, no hardcoded secrets, logging not `print`, no magic strings
- [ ] Semantic version bumped in the affected `pyproject.toml`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Commit message follows Conventional Commits with AUDIT block

## Test plan

<!-- How did you verify this works? What edge cases did you check? -->

## Notes for reviewer

<!-- Anything that needs extra attention, known limitations, or follow-up issues -->
