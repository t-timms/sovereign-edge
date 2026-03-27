"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Global settings. Load from environment or .env file."""

    # Paths — override via SE_PROJECT_ROOT, SE_SSD_ROOT, etc. if an SSD is available
    project_root: Path = Path.home() / "sovereign-edge"
    ssd_root: Path = Path.home() / "sovereign-edge" / "data"
    lancedb_path: Path = Path.home() / "sovereign-edge" / "data" / "lancedb"
    logs_path: Path = Path.home() / "sovereign-edge" / "data" / "logs"
    models_path: Path = Path.home() / "sovereign-edge" / "data" / "models"

    # Ollama
    ollama_host: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    local_llm_model: str = "qwen3:0.6b"

    # Cloud API Keys (loaded from SOPS-decrypted env vars)
    groq_api_key: str = ""
    google_api_key: str = ""
    cerebras_api_key: str = ""
    mistral_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_owner_chat_id: str = ""
    discord_bot_token: str = ""
    discord_owner_user_id: str = ""
    alpha_vantage_key: str = ""
    jina_api_key: str = ""

    # Cloud API Rate Limits (requests per minute) — tune per deployment
    groq_rpm: int = 30
    gemini_rpm: int = 15
    cerebras_rpm: int = 30
    mistral_rpm: int = 2

    # Career Squad — override to target a different location/role
    career_target_location: str = "your city or region"
    career_target_roles: str = "ML Engineer, AI Engineer, LLM Engineer"
    career_differentiators: str = ""  # comma-separated; empty = generic coaching

    # Feature Flags
    voice_enabled: bool = False
    creative_enabled: bool = True
    debug_mode: bool = False

    # Scheduling
    morning_wake_hour: int = 5
    morning_wake_minute: int = 0
    timezone: str = "US/Central"

    model_config = {"env_prefix": "SE_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()


def log_startup_warnings() -> None:
    """Emit warnings for any missing API keys so the operator knows at boot time."""
    s = get_settings()

    cloud_keys = {
        "SE_GROQ_API_KEY": s.groq_api_key,
        "SE_GOOGLE_API_KEY": s.google_api_key,
        "SE_CEREBRAS_API_KEY": s.cerebras_api_key,
        "SE_MISTRAL_API_KEY": s.mistral_api_key,
    }
    required_keys = {
        "SE_TELEGRAM_BOT_TOKEN": s.telegram_bot_token,
        "SE_TELEGRAM_OWNER_CHAT_ID": s.telegram_owner_chat_id,
    }

    for key, value in required_keys.items():
        if not value:
            logger.error("startup_missing_required key=%s — bot will not function", key)

    # Validate chat_id is a valid integer — string comparison fails silently otherwise
    if s.telegram_owner_chat_id and not s.telegram_owner_chat_id.lstrip("-").isdigit():
        logger.error(
            "startup_invalid_chat_id value=%r — must be a numeric Telegram chat ID; "
            "all requests will be rejected",
            s.telegram_owner_chat_id,
        )

    missing_cloud = [k for k, v in cloud_keys.items() if not v]
    for key in missing_cloud:
        logger.warning("startup_missing_cloud_key key=%s — provider skipped", key)

    if len(missing_cloud) == len(cloud_keys):
        logger.warning(
            "startup_no_cloud_providers — all requests will use local Ollama (%s)",
            s.local_llm_model,
        )

    if not s.jina_api_key:
        logger.info(
            "startup_no_jina_key — web search limited to ~200 RPD free tier; "
            "set SE_JINA_API_KEY for unlimited"
        )
