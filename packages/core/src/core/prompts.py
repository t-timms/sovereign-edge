"""Versioned prompt loader for expert agents.

Loads system prompts from YAML files in the prompts/ directory.
Supports semver versioning — each expert can have multiple prompt versions
and the loader picks the latest or a specific version.

Usage:
    from core.prompts import load_prompt, get_latest_prompt

    # Load specific version
    prompt = load_prompt("spiritual", "1.0.0")
    print(prompt["system_prompt"])

    # Load latest version
    prompt = get_latest_prompt("intelligence")
    print(prompt["morning_prompt"])
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from core.config import get_settings

logger = logging.getLogger(__name__)

# Regex for semver: v?MAJOR.MINOR.PATCH
_SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def _parse_semver(version: str) -> tuple[int, int, int] | None:
    """Parse a semver string into (major, minor, patch) or None."""
    match = _SEMVER_RE.match(version)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def _prompts_dir() -> Path:
    """Resolve the prompts directory relative to project root."""
    settings = get_settings()
    return settings.project_root / "prompts"


def list_versions(expert: str) -> list[str]:
    """List all available prompt versions for an expert, sorted ascending."""
    expert_dir = _prompts_dir() / expert
    if not expert_dir.exists():
        return []

    versions: list[tuple[tuple[int, int, int], str]] = []
    for f in expert_dir.glob("v*.yaml"):
        stem = f.stem  # e.g. "v1.0.0"
        parsed = _parse_semver(stem)
        if parsed:
            versions.append((parsed, stem))

    versions.sort(key=lambda x: x[0])
    return [v[1] for v in versions]


@lru_cache(maxsize=32)
def load_prompt(expert: str, version: str) -> dict[str, Any]:
    """Load a specific prompt version for an expert.

    Args:
        expert: Expert name (e.g. "spiritual", "intelligence")
        version: Semver string (e.g. "v1.0.0" or "1.0.0")

    Returns:
        Dict with keys: version, expert, system_prompt, and any additional prompts

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    if not version.startswith("v"):
        version = f"v{version}"

    prompt_path = _prompts_dir() / expert / f"{version}.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with open(prompt_path) as f:
        data = yaml.safe_load(f)

    logger.debug("prompt_loaded expert=%s version=%s", expert, version)
    return data


def get_latest_prompt(expert: str) -> dict[str, Any]:
    """Load the latest prompt version for an expert.

    Returns:
        Dict with prompt data

    Raises:
        FileNotFoundError: If no prompts exist for the expert
    """
    versions = list_versions(expert)
    if not versions:
        raise FileNotFoundError(f"No prompts found for expert: {expert}")

    return load_prompt(expert, versions[-1])


def get_system_prompt(expert: str, version: str | None = None) -> str:
    """Convenience: get just the system_prompt text.

    Args:
        expert: Expert name
        version: Optional specific version (default: latest)

    Returns:
        The system prompt string
    """
    if version:
        data = load_prompt(expert, version)
    else:
        data = get_latest_prompt(expert)
    return data.get("system_prompt", "")


def get_prompt_field(expert: str, field: str, version: str | None = None) -> str:
    """Get a specific prompt field (e.g. "morning_prompt", "devotional_prompt").

    Args:
        expert: Expert name
        field: Field name in the YAML file
        version: Optional specific version (default: latest)

    Returns:
        The prompt field value, or empty string if not found
    """
    if version:
        data = load_prompt(expert, version)
    else:
        data = get_latest_prompt(expert)
    return data.get(field, "")
