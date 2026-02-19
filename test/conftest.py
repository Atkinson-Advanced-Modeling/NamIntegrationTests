"""Pytest configuration and fixtures."""

import json
from pathlib import Path

import pytest


def _load_demonet_config() -> dict:
    """Load demonet config (mirrors nam_full_configs/models/demonet.json)."""
    path = Path(__file__).resolve().parents[1] / "configs" / "demonet.json"
    data = json.loads(path.read_text())
    data.pop("_notes", None)
    data.pop("_comments", None)
    return data


@pytest.fixture
def demonet_config():
    """Demonet config for building WaveNet models."""
    return _load_demonet_config()
