# Troubleshooting

Common errors encountered during deployment and development, with root causes and fixes.

---

## Table of Contents

- [Bot / Startup](#bot--startup)
- [Intent Router](#intent-router)
- [LLM Gateway](#llm-gateway)
- [Expert Agents](#expert-agents)
- [Secrets and Configuration](#secrets-and-configuration)
- [Systemd Service](#systemd-service)

---

## Bot / Startup

### Two bot instances running simultaneously

**Symptom:** Bot responds twice to each message, or you see two `telegram_bot_started` log events.

**Cause:** A second invocation started before the first exited (e.g., manual run while the service is active).

**Fix:** The bot has a built-in single-instance guard via `fcntl.flock`. A second invocation exits immediately with:
```
sovereign-edge-telegram already running (lock held: /tmp/sovereign-edge-telegram.lock). Exiting.
```
If you're seeing double responses despite this, check whether two separate bot tokens are configured pointing to the same chat, or whether the lock file is on a different filesystem than expected.

---

### `No module named 'mem0'`

**Symptom:** Startup log shows `Mem0 initialization failed, episodic memory disabled`.

**Cause:** `mem0` is an optional dependency not installed in the venv.

**Fix:** Either install it or leave it — the bot functions without episodic memory, falling back to conversation history only:

```bash
uv add mem0
```

On Jetson (aarch64), `mem0` may require additional build dependencies. If installation fails, leaving it uninstalled is safe.

---

### `None of PyTorch, TensorFlow, or Flax have been found`

**Symptom:** Logged at startup by the `transformers` library.

**Cause:** `transformers` is installed for the tokenizer but PyTorch is not in the venv. This is expected on the Jetson production venv (PyTorch is managed by Conda/JetPack, not uv).

**Fix:** None needed. The tokenizer loads via the HuggingFace `tokenizers` fast tokenizer backend, which does not require PyTorch. The ONNX router loads and classifies correctly.

---

## Intent Router

### `RevisionNotFoundError` for DistilBERT tokenizer

**Symptom:**
```
RevisionNotFoundError: 26bc1ad6c0ac742e9b52263c5f3d6fc869352be4 is not a valid git identifier
```

**Cause:** A hardcoded git revision hash was used for the HuggingFace tokenizer — the commit no longer exists on the Hub.

**Fix (applied in v0.3.2):** The tokenizer now loads from `data/models/tokenizer/` if the directory exists, falling back to `distilbert-base-uncased` from HuggingFace. Copy the tokenizer files to the Jetson:

```bash
scp -r data/models/tokenizer/ user@jetson:~/sovereign-edge/data/models/
```

---

### ONNX model not loading — keyword fallback active

**Symptom:** Log shows `onnx_model_not_found — using keyword classifier (bootstrap mode)`.

**Cause:** `data/models/router.onnx` does not exist. The ONNX model requires training via `scripts/train-router.py`.

**Fix:** Either train the model (see [scripts/train-router.py](../scripts/train-router.py)) or leave the keyword classifier active. Keyword mode provides ~85% accuracy for development.

```bash
# Check if the model exists
ls data/models/router.onnx

# Copy from development machine if trained
scp data/models/router.onnx user@jetson:~/sovereign-edge/data/models/
scp -r data/models/tokenizer/ user@jetson:~/sovereign-edge/data/models/
```

---

### `CUDAExecutionProvider` not available warning

**Symptom:**
```
UserWarning: Specified provider 'CUDAExecutionProvider' is not in available providers.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```

**Cause:** `onnxruntime` was installed without GPU support, or the CUDA driver is not accessible from within the venv.

**Fix:** None required. ONNX falls back to `CPUExecutionProvider` automatically. The DistilBERT INT8 model is fast enough on CPU (<10ms on Jetson). If you need GPU inference, install `onnxruntime-gpu` instead:

```bash
uv add onnxruntime-gpu
uv remove onnxruntime
```

---

## LLM Gateway

### Groq `tool_use_failed` on structured calls

**Symptom:**
```
structured_provider_failed model=groq/meta-llama/llama-4-scout-17b-16e-instruct
AssertionError: Instructor does not support multiple tool calls
```

**Cause:** `llama-4-scout-17b` generates free-text responses instead of JSON tool calls for complex prompts. `instructor`'s `parse_tools` assertion fails because `tool_calls` is `None`.

**Fix (applied in v0.3.2):** Groq is marked `supports_structured=False` in `gateway.py`. It is skipped for all `complete_structured()` calls and only used for unstructured `complete()` and streaming.

---

### Gemini returning `choices=[]`

**Symptom:**
```
empty_response_from provider=gemini/gemini-2.5-flash
```

**Cause:** Gemini's safety filter is blocking large prompts (typically >6000 tokens for intelligence queries). The model returns an empty choices array rather than an error.

**Fix:** The gateway handles this by treating an empty response as a provider failure and moving to the next provider. No configuration change needed. If this happens frequently for intelligence queries, consider reducing the system prompt length or the number of papers included in the context.

---

### Cerebras 404 on structured calls

**Symptom:**
```
structured_provider_failed model=cerebras/llama3.3-70b
```

**Cause:** The model `llama3.3-70b` is not available on the Cerebras API (returns 404 regardless of name format).

**Fix (applied in v0.3.3):** Cerebras is marked `supports_structured=False`. It is still in the unstructured chain but fast-fails. Structured chain is now Gemini → Mistral only (with Gemini preferred).

---

### All structured providers failing — response is 194 chars

**Symptom:** Career responses are very short; logs show both Gemini and Mistral failing structured calls.

**Cause chain:**
1. Jina search returns 422 → no search results in context
2. Gemini may be rate-limited → skipped silently
3. Mistral returns free-text instead of tool calls → `instructor` fails
4. Fallback: `gateway.complete()` with Groq, short response on empty context

**Fix:** Jina 422 was the root cause — with live search results, the LLM has enough context to produce a quality response. Confirm Jina searches are succeeding:
```bash
journalctl -u telegram-bot | grep jina_search
```

---

## Expert Agents

### Career: `jina_search_http_error status=422`

**Symptom:**
```
jina_search_http_error status=422 query='(site:greenhouse.io OR ...) ...' attempt=1
```

**Cause:** Jina's search API is a semantic search endpoint — it does not support Google-style `site:` operators, `OR` chains, or `-word` exclusions. These return 422 Unprocessable Entity.

**Fix (applied in v0.3.2):** Career subgraph now uses plain natural-language queries without operators. If you see 422 on custom queries, remove any Google search syntax.

---

### Spiritual: `bible_lookup_failed reference='prov'`

**Symptom:**
```
bible_lookup_failed reference='prov' attempt=1 error=Client error '404 Not Found'
```

**Cause:** `bible-api.com` does not accept bare book names without a chapter number. The regex matched "prov" as a valid reference (the chapter/verse group is optional in the pattern).

**Fix (applied in v0.3.2):** `extract_reference()` now requires at least one digit in the matched text. Bare book names fall through to `random_verse()` instead of triggering a lookup.

---

### Flashrank 404 on model download

**Symptom:**
```
flashrank_model_download_failed ms-marco-MiniLM-L-4-v2.zip 404
```

**Cause:** The `ms-marco-MiniLM-L-4-v2` flashrank model is no longer available at the expected URL. The intelligence ranker falls back to keyword ranking automatically.

**Fix:** The keyword fallback is functional. To restore neural ranking, update the flashrank model name in the intelligence expert configuration to a currently available model, or install the model files manually.

---

## Secrets and Configuration

### Old bot token still being used after sops update

**Symptom:** Bot responds with the old token; `telegram_bot_started` shows old bot identity.

**Cause:** `sops --set` was run with the wrong key name, or the decrypted `.env` was not refreshed before restarting the service.

**Fix:**

```bash
# Verify the correct key name in the encrypted file
sops -d secrets/env.yaml | grep TELEGRAM

# Update using the exact key from the file
sops --set '["se_telegram_bot_token"] "new_token_here"' secrets/env.yaml

# Restart to pick up the new secret
sudo systemctl restart telegram-bot.service
```

---

### `sops: error: decryption failed`

**Cause:** The Age private key at `~/.config/sops/age/keys.txt` does not match the public key used to encrypt `secrets/env.yaml`.

**Fix:** Ensure the Age key file is present on the device:

```bash
ls ~/.config/sops/age/keys.txt
```

If deploying to a new device, re-encrypt `secrets/env.yaml` with the new device's public key, or transfer the Age key file securely to the device.

---

## Systemd Service

### Service starts then immediately stops

**Symptom:** `sudo systemctl status telegram-bot` shows `Active: failed` seconds after start.

**Common causes:**

1. **SOPS decryption failed** — check `journalctl -u telegram-bot -n 20` for `sops: error` messages
2. **Python venv not found** — verify `ExecStart` path in the service file matches your actual install location
3. **Lock file held** — another process holds `/tmp/sovereign-edge-telegram.lock`. Check with `fuser /tmp/sovereign-edge-telegram.lock`

```bash
# View last 50 lines of service log
journalctl -u telegram-bot -n 50

# Check if another instance is running
fuser /tmp/sovereign-edge-telegram.lock
ps aux | grep telegram_bot
```

---

### `No space left on device` in logs

**Cause:** SQLite databases or LanceDB files have grown to fill the storage partition.

**Fix:**

```bash
# Check storage usage
df -h
du -sh ~/sovereign-edge/data/*

# Compact the skill library SQLite (WAL mode accumulates -wal files)
sqlite3 data/skills.db "VACUUM;"
sqlite3 data/traces.db "VACUUM;"
```

Consider mounting `data/` on a separate SSD and setting `SE_SSD_ROOT` in your secrets file.
