from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def restore_config_defaults(monkeypatch):
    yield
    monkeypatch.delenv("ATOMIC_AGENT_CALL_TIMEOUT", raising=False)
    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])


def test_call_timeout_defaults_to_300_seconds(monkeypatch):
    monkeypatch.delenv("ATOMIC_AGENT_CALL_TIMEOUT", raising=False)

    import config

    importlib.reload(config)

    assert config.CALL_TIMEOUT_SEC == 300


def test_call_timeout_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_CALL_TIMEOUT", "240")

    import config

    importlib.reload(config)

    assert config.CALL_TIMEOUT_SEC == 240
