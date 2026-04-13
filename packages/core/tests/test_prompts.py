"""Tests for the versioned prompt loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from core.prompts import (
    _parse_semver,
    get_latest_prompt,
    get_prompt_field,
    get_system_prompt,
    list_versions,
    load_prompt,
)


class TestSemverParsing:
    def test_valid_semver(self) -> None:
        assert _parse_semver("1.0.0") == (1, 0, 0)
        assert _parse_semver("v1.0.0") == (1, 0, 0)
        assert _parse_semver("v2.3.1") == (2, 3, 1)

    def test_invalid_semver(self) -> None:
        assert _parse_semver("abc") is None
        assert _parse_semver("1.0") is None
        assert _parse_semver("") is None


class TestPromptLoader:
    @pytest.fixture()
    def prompt_dir(self, tmp_path: Path) -> Path:
        """Create a temporary prompts directory with test data."""
        expert_dir = tmp_path / "prompts" / "test_expert"
        expert_dir.mkdir(parents=True)

        v1 = {
            "version": "1.0.0",
            "expert": "test_expert",
            "system_prompt": "You are test v1.",
            "morning_prompt": "Morning v1.",
        }
        v2 = {
            "version": "1.1.0",
            "expert": "test_expert",
            "system_prompt": "You are test v1.1.",
            "extra_field": "bonus",
        }

        (expert_dir / "v1.0.0.yaml").write_text(yaml.dump(v1))
        (expert_dir / "v1.1.0.yaml").write_text(yaml.dump(v2))

        return tmp_path / "prompts"

    def test_list_versions(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            versions = list_versions("test_expert")
        assert versions == ["v1.0.0", "v1.1.0"]

    def test_list_versions_nonexistent(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            versions = list_versions("nonexistent")
        assert versions == []

    def test_load_specific_version(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            load_prompt.cache_clear()
            data = load_prompt("test_expert", "v1.0.0")
        assert data["system_prompt"] == "You are test v1."
        assert data["morning_prompt"] == "Morning v1."

    def test_load_without_v_prefix(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            load_prompt.cache_clear()
            data = load_prompt("test_expert", "1.0.0")
        assert data["version"] == "1.0.0"

    def test_load_nonexistent_raises(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            load_prompt.cache_clear()
            with pytest.raises(FileNotFoundError):
                load_prompt("test_expert", "v9.9.9")

    def test_get_latest(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            load_prompt.cache_clear()
            data = get_latest_prompt("test_expert")
        assert data["version"] == "1.1.0"
        assert data["system_prompt"] == "You are test v1.1."

    def test_get_system_prompt(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            load_prompt.cache_clear()
            text = get_system_prompt("test_expert")
        assert "test v1.1" in text

    def test_get_prompt_field(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            load_prompt.cache_clear()
            val = get_prompt_field("test_expert", "extra_field")
        assert val == "bonus"

    def test_get_missing_field_returns_empty(self, prompt_dir: Path) -> None:
        with patch("core.prompts._prompts_dir", return_value=prompt_dir):
            load_prompt.cache_clear()
            val = get_prompt_field("test_expert", "nonexistent")
        assert val == ""
