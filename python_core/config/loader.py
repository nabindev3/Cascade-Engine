"""
Configuration loader — reads YAML config and merges with environment variables.
"""

import os
from pathlib import Path
from typing import Any

# Use pyyaml if available, fall back to a simple parser
try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG = {
    "engines": {
        "local": {
            "enabled": True,
            "base_url": "http://localhost:11434",
            "model": "llama3.2:3b",
            "timeout_s": 30.0,
            "cost_per_token": 0.000001,
        },
        "mid": {
            "enabled": True,
            "engine_id": "openai-mid",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "timeout_s": 45.0,
            "cost_per_input_token": 0.00000015,
            "cost_per_output_token": 0.0000006,
        },
        "premium": {
            "enabled": True,
            "engine_id": "openai-premium",
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "timeout_s": 60.0,
            "cost_per_input_token": 0.0000025,
            "cost_per_output_token": 0.00001,
        },
    },
    "router": {
        "confidence_thresholds": {1: 0.65, 2: 0.80},
        "max_cost_per_request": 0.05,
        "reliability_ema_alpha": 0.1,
        "min_reliability_to_attempt": 0.3,
    },
    "logging": {
        "output_dir": "./data/logs",
    },
}


def load_config(config_path: str = None) -> dict:
    """
    Load configuration with priority:
    1. Environment variables (highest)
    2. Config file (YAML)
    3. Default config (lowest)
    """
    config = _deep_copy(DEFAULT_CONFIG)

    # Load from YAML file if it exists
    if config_path is None:
        config_path = os.environ.get("CASCADE_CONFIG", "./config.yaml")

    path = Path(config_path)
    if path.exists() and yaml:
        with open(path) as f:
            file_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, file_config)

    # Override with environment variables
    env_overrides = {
        "CASCADE_LOCAL_URL": ("engines", "local", "base_url"),
        "CASCADE_LOCAL_MODEL": ("engines", "local", "model"),
        "CASCADE_MID_API_KEY": ("engines", "mid", "api_key"),
        "CASCADE_MID_MODEL": ("engines", "mid", "model"),
        "CASCADE_MID_BASE_URL": ("engines", "mid", "base_url"),
        "CASCADE_PREMIUM_API_KEY": ("engines", "premium", "api_key"),
        "CASCADE_PREMIUM_MODEL": ("engines", "premium", "model"),
        "CASCADE_PREMIUM_BASE_URL": ("engines", "premium", "base_url"),
        "CASCADE_LOG_DIR": ("logging", "output_dir"),
        "CASCADE_MAX_COST": ("router", "max_cost_per_request"),
    }

    for env_var, key_path in env_overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            _set_nested(config, key_path, _coerce_value(value))

    return config


def _deep_copy(d: dict) -> dict:
    """Simple deep copy for nested dicts."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy(v)
        elif isinstance(v, list):
            result[k] = v[:]
        else:
            result[k] = v
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = _deep_copy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_nested(d: dict, keys: tuple, value: Any):
    """Set a value in a nested dict by key path."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _coerce_value(value: str) -> Any:
    """Try to coerce string env vars to appropriate types."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value
