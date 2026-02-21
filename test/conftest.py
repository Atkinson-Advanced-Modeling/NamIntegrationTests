"""Pytest configuration and fixtures."""

import pytest as _pytest

from _configs import load_demonet_config


@_pytest.fixture
def demonet_config():
    """Demonet config for building WaveNet models."""
    return load_demonet_config()
