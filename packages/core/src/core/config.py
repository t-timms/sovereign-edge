"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
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

    # Cloud API Keys (loaded from SOPS-decrypted env vars) — SecretStr prevents accidental logging
    groq_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")  # SE_GEMINI_API_KEY
    google_api_key: SecretStr = SecretStr("")  # legacy alias — prefer SE_GEMINI_API_KEY
    cerebras_api_key: SecretStr = SecretStr("")
    mistral_api_key: SecretStr = SecretStr("")
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_owner_chat_id: str = ""
    discord_bot_token: SecretStr = SecretStr("")
    discord_owner_user_id: str = ""
    jina_api_key: SecretStr = SecretStr("")
    fmp_api_key: SecretStr = SecretStr("")  # Financial Modeling Prep — earnings transcripts

    # Observability and runtime
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
    mem0_user_id: str = "john"  # Mem0 episodic memory namespace

    # Router / classifier
    router_confidence_threshold: float = 0.7
    router_model_path: Path | None = None  # path to a .onnx router model; None = keyword fallback

    # Voice service
    stt_model: str = "base.en"  # faster-whisper: tiny.en, base.en, small.en, medium.en

    # Cloud API Rate Limits (requests per minute) — tune per deployment
    groq_rpm: int = 30
    gemini_rpm: int = 15
    cerebras_rpm: int = 30
    mistral_rpm: int = 2

    @field_validator("groq_rpm", "gemini_rpm", "cerebras_rpm", "mistral_rpm")
    @classmethod
    def _rpm_must_be_positive(cls, v: int) -> int:
        if v < 1:
            msg = f"RPM must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("log_level")
    @classmethod
    def _log_level_valid(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            msg = f"log_level must be one of {sorted(valid)}, got {v!r}"
            raise ValueError(msg)
        return v.upper()

    def active_llm_providers(self) -> list[str]:
        """Return litellm model strings for providers that have API keys configured.

        Returned in priority order: Groq → Gemini → Cerebras → Mistral.
        """
        providers: list[str] = []
        if self.groq_api_key.get_secret_value():
            providers.append("groq/meta-llama/llama-4-scout-17b-16e-instruct")
        gem_key = self.gemini_api_key.get_secret_value() or self.google_api_key.get_secret_value()
        if gem_key:
            providers.append("gemini/gemini-2.5-flash")
        if self.cerebras_api_key.get_secret_value():
            providers.append("cerebras/llama-3.3-70b")
        if self.mistral_api_key.get_secret_value():
            providers.append("mistral/mistral-small-latest")
        return providers

    def has_router_model(self) -> bool:
        """Return True if a trained router .onnx model exists on disk."""
        if self.router_model_path is None:
            return False
        return Path(self.router_model_path).exists()

    # Career Expert — override to target a different location/role
    career_target_location: str = "Dallas Fort Worth TX"
    career_target_cities: str = "Dallas,Fort Worth,Plano,Irving,Frisco,Allen,McKinney,Richardson,Arlington,Southlake,Addison,Carrollton"  # noqa: E501
    career_target_roles: str = (
        "ML Engineer, AI Engineer, Data Scientist, Data Engineer, Data Analyst, MLOps Engineer"
    )
    career_differentiators: str = ""  # comma-separated; empty = generic coaching

    # Career Expert — job intelligence and resume matching
    # Path to folder containing resume PDFs (used for skill extraction + job matching)
    career_resume_path: Path = Path.home() / "Documents" / "Job Search" / "Resumes"
    # SQLite DB for job deduplication. None = use ssd_root/jobs.db
    career_job_db_path: Path | None = None
    # Don't resurface jobs seen within this many days
    career_dedup_window_days: int = 7
    # Adzuna free job API — sign up at developer.adzuna.com (50 req/day free)
    adzuna_app_id: SecretStr = SecretStr("")  # SE_ADZUNA_APP_ID
    adzuna_app_key: SecretStr = SecretStr("")  # SE_ADZUNA_APP_KEY

    # Intelligence Expert — comma-separated repo topics for paper relevance scoring
    # e.g. "bible-ai:rag,orpo,fine-tuning,graphrag; sovereign-edge:langgraph,agents,mcp; gpu-suite:inference,tensorrt,vllm"  # noqa: E501
    repo_topics: str = "bible-ai:rag,orpo,fine-tuning,graphrag,retrieval; sovereign-edge:langgraph,agents,mcp,tool-use; gpu-suite:inference,tensorrt,vllm,exllamav2,quantization,benchmark,cuda"  # noqa: E501
    # Market data watchlist and alert threshold
    watchlist: list[str] = []  # e.g. ["NVDA", "MSFT", "AAPL"]
    market_alert_threshold: float = 0.02  # 2% price move triggers an alert

    # Goals Agent
    goals_enabled: bool = True
    goals_db_path: Path | None = None  # defaults to ssd_root/goals.db

    # Web Dashboard (served on the health port)
    dashboard_token: SecretStr = SecretStr("")  # set SE_DASHBOARD_TOKEN to enable

    # WhatsApp via Twilio
    whatsapp_enabled: bool = False
    twilio_account_sid: SecretStr = SecretStr("")
    twilio_auth_token: SecretStr = SecretStr("")
    twilio_whatsapp_from: str = ""  # E.164 format, no "whatsapp:" prefix — e.g. "+14155238886"
    whatsapp_owner_number: str = ""  # E.164 format, no "whatsapp:" prefix — e.g. "+12125551234"

    # MCP Tool Server
    mcp_enabled: bool = False
    mcp_port: int = 3000

    # Feature Flags
    voice_enabled: bool = False
    creative_enabled: bool = True
    debug_mode: bool = False
    # Self-improvement: run a quality critique pass after each CLOUD expert response.
    # Adds ~1-2 s latency per call. Set SE_REFLECT_ENABLED=true to enable.
    reflect_enabled: bool = False

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
        "SE_GROQ_API_KEY": s.groq_api_key.get_secret_value(),
        "SE_GEMINI_API_KEY": (
            s.gemini_api_key.get_secret_value() or s.google_api_key.get_secret_value()
        ),
        "SE_CEREBRAS_API_KEY": s.cerebras_api_key.get_secret_value(),
        "SE_MISTRAL_API_KEY": s.mistral_api_key.get_secret_value(),
    }
    required_keys = {
        "SE_TELEGRAM_BOT_TOKEN": s.telegram_bot_token.get_secret_value(),
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

    if not s.jina_api_key.get_secret_value():
        logger.info(
            "startup_no_jina_key — web search limited to ~200 RPD free tier; "
            "set SE_JINA_API_KEY for unlimited"
        )
    if not s.adzuna_app_id.get_secret_value():
        logger.info(
            "startup_no_adzuna_key — Adzuna job source disabled; "
            "free signup at developer.adzuna.com then set SE_ADZUNA_APP_ID + SE_ADZUNA_APP_KEY"
        )
    if not s.career_resume_path.exists():
        logger.warning(
            "startup_resume_path_missing path=%s — resume skill matching disabled; "
            "set SE_CAREER_RESUME_PATH to your resumes folder",
            s.career_resume_path,
        )
