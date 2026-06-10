from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest




@pytest.fixture(autouse=True)
def restore_config_defaults(monkeypatch):
    yield
    monkeypatch.delenv("ATOMIC_AGENT_CALL_TIMEOUT", raising=False)
    if "generative.config" in sys.modules:
        importlib.reload(sys.modules["generative.config"])


def test_call_timeout_defaults_to_300_seconds(monkeypatch):
    monkeypatch.delenv("ATOMIC_AGENT_CALL_TIMEOUT", raising=False)

    from generative import config

    importlib.reload(config)

    assert config.CALL_TIMEOUT_SEC == 300


def test_call_timeout_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_CALL_TIMEOUT", "240")

    from generative import config

    importlib.reload(config)

    assert config.CALL_TIMEOUT_SEC == 240
