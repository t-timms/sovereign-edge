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

## Out of Scope

- Issues that require physical access to the Jetson device
- Issues in upstream dependencies (report those to the respective project)
- The Telegram or Discord platforms themselves
