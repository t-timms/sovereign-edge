# Deployment

Sovereign Edge runs as a systemd service on a Jetson Orin or any Linux ARM64/x86 host with Ollama. This document covers the full setup from a fresh board to a running service.

---

## Prerequisites

- NVIDIA Jetson Orin (JetPack 6, Ubuntu 22.04) or any Linux ARM64/x86 host
- Python 3.11+
- `uv` package manager
- Ollama installed and running
- SOPS + Age for secrets management
- `rsync` on the development machine (for deployment)

> **LangGraph:** `uv sync --all-packages` installs `langgraph>=1.0` automatically. This is required for expert subgraph pipelines and the director graph. Running `uv sync` without `--all-packages` omits it; the bot functions but falls back to single-step LLM calls per expert.

---

## First-Time Setup

### 1. Install system dependencies

```bash
sudo apt update && sudo apt install -y \
    python3.11 python3.11-venv \
    curl git rsync

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

### 3. Pull the required models

```bash
# Chat model (local fallback)
ollama pull qwen3:0.6b

# Embedding model (intent routing Tier 1)
ollama pull qwen3-embedding:0.6b
```

### 4. Clone the repository

```bash
# Replace placeholders with your Linux username and repo path
git clone https://github.com/t-timms/sovereign-edge.git ~/sovereign-edge
cd ~/sovereign-edge
```

### 5. Install Python dependencies

```bash
uv sync --all-packages
```

### 6. Create the data directory

```bash
mkdir -p ~/sovereign-edge/data/{lancedb,logs,models}
# Or on a separate SSD:
# mkdir -p /mnt/ssd/sovereign-edge-data/{lancedb,logs,models}
# Then set SE_SSD_ROOT=/mnt/ssd/sovereign-edge-data in your secrets file
```

---

## Secrets Management

Sovereign Edge uses SOPS with Age encryption to store API keys. Keys are never stored in plaintext in the repository.

### Install SOPS and Age

```bash
# Age
sudo apt install age

# SOPS
curl -LO https://github.com/getsops/sops/releases/latest/download/sops-v3.x.x.linux.arm64
sudo install sops-v3.x.x.linux.arm64 /usr/local/bin/sops
```

### Generate an Age key pair

```bash
mkdir -p ~/.config/sops/age
age-keygen -o ~/.config/sops/age/keys.txt
```

Copy the public key from the output (starts with `age1...`).

### Create the encrypted secrets file

On your development machine, create `secrets/env.yaml`:

```yaml
SE_TELEGRAM_BOT_TOKEN: <your_bot_token>
SE_TELEGRAM_OWNER_CHAT_ID: "<your_numeric_chat_id>"
SE_GROQ_API_KEY: <gsk_...>
SE_GOOGLE_API_KEY: <AIza...>
SE_CEREBRAS_API_KEY: <csk_...>
SE_MISTRAL_API_KEY: <your_mistral_key>
SE_SSD_ROOT: /mnt/ssd/sovereign-edge-data
```

Encrypt it with your public key:

```bash
sops --encrypt --age <age1_your_public_key> secrets/env.yaml > secrets/env.yaml.enc
mv secrets/env.yaml.enc secrets/env.yaml
```

The encrypted `secrets/env.yaml` is safe to commit. The plaintext version is not.

### `.sops.yaml` configuration

Create `.sops.yaml` at the project root to configure key discovery:

```yaml
creation_rules:
  - path_regex: secrets/.*\.yaml$
    age: <age1_your_public_key>
```

---

## systemd Service

The service file is at `systemd/telegram-bot.service`. It uses `<DEPLOY_USER>` and `<DEPLOY_ROOT>` placeholders that must be substituted with your values before installing.

### Configure and install the service

```bash
# Set your values
DEPLOY_USER=$(whoami)
DEPLOY_ROOT=$(realpath ~/sovereign-edge)

# Substitute placeholders and install
sed "s|<DEPLOY_USER>|${DEPLOY_USER}|g; s|<DEPLOY_ROOT>|${DEPLOY_ROOT}|g" \
    systemd/telegram-bot.service \
    | sudo tee /etc/systemd/system/telegram-bot.service

sudo systemctl daemon-reload
sudo systemctl enable telegram-bot
```

### Service configuration highlights

The service:
- Decrypts `secrets/env.yaml` via SOPS on every start — fails hard if SOPS errors, preventing the bot from starting with no secrets
- Runs under your user account (non-root)
- Enforces memory and CPU limits (tune `MemoryMax` and `CPUQuota` for your hardware)
- Applies systemd hardening: `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem`

### Service management

```bash
# Start / stop / restart
sudo systemctl start telegram-bot
sudo systemctl stop telegram-bot
sudo systemctl restart telegram-bot

# View status
sudo systemctl status telegram-bot

# Follow logs
journalctl -u telegram-bot -f

# View today's logs
journalctl -u telegram-bot --since today
```

---

## Deployment from Development Machine

The project includes a `Taskfile.yml` that automates deploy + restart over SSH.

### Configure deployment targets

Export these environment variables (add to your shell profile):

```bash
export DEPLOY_HOST=<your-device-ip-or-hostname>   # Tailscale IP, local IP, or hostname
export DEPLOY_USER=<your_linux_username>           # Linux username on the target device
```

Then deploy:

```bash
task deploy
```

This runs `rsync` to sync the project directory to the device (excluding `.git`, `__pycache__`, and secrets), then SSHs in to restart the service.

---

## Verifying the Deployment

1. Send `/start` to your bot in Telegram — you should receive the welcome message.
2. Send `/stats` to confirm the trace store is recording.
3. Send a test message ("What's a good paper on LLM inference?") — verify the intelligence expert responds with a properly formatted reply.
4. Confirm LangGraph subgraph pipelines are active by checking the startup logs for `"intelligence_expert"`, `"career_expert"`, `"creative_expert"`, and `"spiritual_expert"` compilation messages. If LangGraph is not installed, experts will log a warning and fall back to direct LLM calls — the bot remains functional but single-step only.
5. Check logs for structured JSON output:

```bash
journalctl -u telegram-bot -f | python3 -m json.tool
```

6. At the configured morning wake time, verify the scheduled brief arrives.

---

## Jetson Orin Nano Performance Tuning

The Jetson Orin Nano (8GB) runs sovereign-edge with Ollama for local inference. These optimizations squeeze maximum performance from the hardware.

### Power Mode & Clock Pinning

The systemd service automatically sets MAXN mode and pins clocks on startup. To do it manually:

```bash
# MAXN power mode — 15W, all 6 cores + full GPU clocks (~20% faster)
sudo nvpmodel -m 0

# Pin clocks to maximum — prevents dynamic downclocking under load
sudo jetson_clocks

# Verify
sudo nvpmodel -q        # should show "NV Power Mode: MAXN"
jetson_clocks --show     # all clocks at max
```

### Memory Management

The Orin Nano has 8GB shared between CPU and GPU. Every MB matters.

```bash
# Disable desktop GUI — saves ~1GB RAM
sudo systemctl set-default multi-user.target
sudo reboot

# Add swap (prevents OOM on larger models)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Enable ZRAM for compressed swap (complementary to disk swap)
sudo apt install zram-config
sudo systemctl enable zram-config
```

### Model Optimization

The default `qwen3:0.6b` underutilizes the hardware. Upgrade path:

| Model | Quantization | VRAM | Tokens/sec (est.) | Quality |
|-------|-------------|------|-------------------|---------|
| qwen3:0.6b | FP16 | ~1.2GB | ~50 tok/s | Baseline |
| qwen3:4b | Q4_K_M | ~3.0GB | ~25-35 tok/s | Much better |
| qwen3:8b | Q4_K_M | ~5.5GB | ~12-18 tok/s | Best quality |
| phi-4-mini | Q4_K_M | ~2.5GB | ~30-40 tok/s | Strong reasoning |

```bash
# Pull a larger quantized model — fits in 8GB with headroom
ollama pull qwen3:4b

# Or for maximum quality (tight fit, needs swap)
ollama pull qwen3:8b
```

**TensorRT-LLM** (advanced — maximum throughput):

```bash
# INT4 weight-only via TensorRT-LLM (JetPack 6 / CUDA 12.2)
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
trtllm-build --model_dir ./qwen3-4b --output_dir ./engine \
    --gemm_plugin auto --weight_only_precision int4
```

### Monitoring

```bash
# Real-time Jetson stats (CPU, GPU, RAM, power, temperature)
tegrastats --interval 1000

# GPU memory specifically
nvidia-smi

# System memory
free -h
```

### Benchmarking

Run the built-in benchmark to compare models on your hardware:

```bash
# Benchmark default model (10 runs)
python scripts/jetson_benchmark.py

# Compare multiple models
python scripts/jetson_benchmark.py --models qwen3:0.6b qwen3:4b --runs 15

# Save results to JSON
python scripts/jetson_benchmark.py --models qwen3:4b --output benchmark_results.json
```

The benchmark reports tokens/sec (mean ± std), latency percentiles (p50/p95/p99), and GPU memory usage.

### Thermal Management

Sustained inference heats the Orin Nano. Without active cooling, thermal throttling kicks in at ~85°C.

- **Stock fan**: Adequate for light use, throttles under sustained load
- **Noctua NF-A4x20 40mm**: ~$15, keeps temps under 65°C at full load
- Mount with thermal tape or 3D-printed bracket on the heatsink

Monitor temperature: `cat /sys/devices/virtual/thermal/thermal_zone*/temp`

---

## ONNX Router Model (Optional)

The Tier 2 ONNX DistilBERT classifier requires a fine-tuned model file. Without it, the router falls through to Tier 3 (keyword matching) automatically.

If you have the model:

```bash
mkdir -p data/models
cp router.onnx data/models/
```

The model path is `{SE_MODELS_PATH}/router.onnx`. The HuggingFace tokenizer is fetched at startup and pinned to a specific commit hash for supply chain security.
