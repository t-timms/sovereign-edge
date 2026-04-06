from __future__ import annotations

import pytest
from core.config import Settings
from pydantic import ValidationError


def test_settings_defaults() -> None:
    s = Settings()
    assert s.log_level == "INFO"
    assert s.router_confidence_threshold == 0.7
    assert s.market_alert_threshold == 0.02
    assert s.mem0_user_id == "john"


def test_settings_invalid_log_level() -> None:
    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")


def test_active_llm_providers_empty_when_no_keys() -> None:
    s = Settings(groq_api_key="", gemini_api_key="", cerebras_api_key="", mistral_api_key="")
    assert s.active_llm_providers() == []


def test_active_llm_providers_priority_order() -> None:
    s = Settings(groq_api_key="gsk_test", gemini_api_key="gem_test")
    providers = s.active_llm_providers()
    assert providers[0].startswith("groq/")
    assert providers[1].startswith("gemini/")


def test_has_router_model_false_when_missing(tmp_path: pytest.fixture) -> None:
    s = Settings(router_model_path=tmp_path / "nonexistent.onnx")
    assert s.has_router_model() is False


def test_has_router_model_true_when_present(tmp_path: pytest.fixture) -> None:
    model = tmp_path / "router.onnx"
    model.write_bytes(b"fake")
    s = Settings(router_model_path=model)
    assert s.has_router_model() is True


def test_get_settings_is_cached() -> None:
    from core.config import get_settings

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
