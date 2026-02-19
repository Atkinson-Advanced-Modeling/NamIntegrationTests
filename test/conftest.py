"""Pytest configuration and fixtures."""

import json as _json
from pathlib import Path as _Path

import pytest as _pytest


def _load_demonet_config() -> dict:
    """Load demonet config (mirrors nam_full_configs/models/demonet.json)."""
    path = _Path(__file__).resolve().parents[1] / "configs" / "demonet.json"
    data = _json.loads(path.read_text())
    data.pop("_notes", None)
    data.pop("_comments", None)
    return data


@_pytest.fixture
def demonet_config():
    """Demonet config for building WaveNet models."""
    return _load_demonet_config()
