# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | ✅        |
| < 0.3.0 | ❌        |

---

## Security Model

Sovereign Edge is a **single-user, single-owner system** — not a multi-tenant service. The security model is designed around that constraint:

- **Access control**: All Telegram and Discord handlers authenticate against a hardcoded owner chat ID / user ID (`SE_TELEGRAM_OWNER_CHAT_ID`, `SE_DISCORD_OWNER_USER_ID`). Messages from any other user are dropped immediately, before any LLM call.
- **PII protection**: The intent router detects PII patterns (SSNs, credit card numbers, phone numbers) and forces all routing to local Ollama inference — no PII ever reaches a cloud provider.
- **Secrets management**: API keys are stored in SOPS-encrypted `secrets/env.yaml` (Age keypair). The plaintext is written to a `chmod 600` file at service startup and never committed to git.
- **Process isolation**: systemd service runs as a non-root user with `NoNewPrivileges=true`, `PrivateTmp=true`, `ProtectSystem=strict`, and `ReadWritePaths` limited to the project root.
- **LLM output validation**: All structured responses are validated through `instructor` + Pydantic schemas — raw LLM JSON is never trusted.
- **Prompt injection**: User input is not injected directly into system prompts. Expert system prompts are static; user content passes through the router and is treated as data, not instructions.

---

## Reporting a Vulnerability

Use [GitHub private security advisories](https://github.com/omnipotence-eth/sovereign-edge/security/advisories/new) to report vulnerabilities privately.

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- The affected version(s)
- Any suggested mitigations

Expect an initial response within 7 days.

---

## Known Past Incidents

### LiteLLM 1.82.7 / 1.82.8 — Supply Chain Compromise (March 2026)

**What happened**: LiteLLM versions 1.82.7 and 1.82.8 were published with a malicious `.claude/settings.json` embedded in the package. When installed, this file could override Claude Code's hooks configuration in the working directory — enabling hook injection attacks that could exfiltrate secrets or execute arbitrary commands via Claude Code's pre/post-tool hooks.

**Affected versions**: `1.82.7`, `1.82.8`

**Fix**: Resolved in `1.83.0` — the file was removed and a semgrep rule was added to block it from being re-introduced. Sovereign Edge pinned at `1.82.6` until 1.83.0 was confirmed clean, then upgraded to `>=1.83.0`.

**Action taken**: This project checked for injected `.claude/settings.json` files (none found). LiteLLM upgraded to `>=1.83.0,<2.0.0` in `packages/llm/pyproject.toml`.

---

## Out of Scope

- Issues that require physical access to the Jetson device
- Issues in upstream dependencies (report those to the respective project)
- The Telegram or Discord platforms themselves
