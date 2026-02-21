"""Loads settings from config.yaml, env vars, and CLI flags."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


class WhisperConfig(BaseModel):
    mode: str = "local"
    model: str = "medium"
    language: str = "es"


class JingleConfig(BaseModel):
    reference_path: str = "assets/jingle_reference.wav"
    threshold: float = 0.6


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3


class SegmentsConfig(BaseModel):
    count: int = 3
    min_seconds: int = 25
    max_seconds: int = 40


class Config(BaseModel):
    whisper: WhisperConfig = WhisperConfig()
    jingle: JingleConfig = JingleConfig()
    llm: LLMConfig = LLMConfig()
    segments: SegmentsConfig = SegmentsConfig()

    # API keys — loaded from env vars, never from config files
    anthropic_api_key: str = ""
    openai_api_key: str = ""


def load_config(
    config_path: Path | None = None,
    cli_overrides: dict | None = None,
) -> Config:
    """Build config: defaults → yaml → env vars → CLI flags."""
    data: dict = {}

    # 1. Load from YAML if it exists
    path = config_path or _DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    # 2. Apply CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is None:
                continue
            # Flat keys like "llm.provider" → nested dict
            parts = key.split(".")
            d = data
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    config = Config(**data)

    # 3. Load API keys from env vars
    config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    return config
