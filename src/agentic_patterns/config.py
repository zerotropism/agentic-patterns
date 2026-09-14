"""Configuration loading, resolved from the project root."""

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration. Raises rather than returning an empty dict."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    config = yaml.safe_load(config_path.read_text())
    if not config:
        raise ValueError(f"Configuration file '{config_path}' is empty.")
    return config


def model_name(config: dict) -> str:
    """AGENTIC_PATTERNS_MODEL wins over config.yaml: comparing models needs no commit."""
    return os.getenv("AGENTIC_PATTERNS_MODEL", config["model"]["name"])
