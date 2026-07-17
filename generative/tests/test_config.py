from __future__ import annotations

import importlib
import sys

import pytest


_MODEL_ENV_VARS = (
    "ATOMIC_AGENT_MODEL_MAIN",
    "ATOMIC_AGENT_MODEL_OPUS",
    "ATOMIC_AGENT_MODEL_JUDGE",
    "ATOMIC_AGENT_MODEL_PLANNER",
    "ATOMIC_AGENT_MODEL_EXTRACTOR",
    "ATOMIC_AGENT_MODEL_EXTENDER",
    "ATOMIC_AGENT_MODEL_CANONICALIZER",
)


@pytest.fixture(autouse=True)
def restore_config_defaults(monkeypatch):
    yield
    monkeypatch.delenv("ATOMIC_AGENT_CALL_TIMEOUT", raising=False)
    for var in _MODEL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
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


# ---------------------------------------------------------------------------
# #317: MODEL_MAIN/MODEL_JUDGE Namensschema + Env-Prioritätsreihenfolge
# ---------------------------------------------------------------------------


def test_model_main_defaults_to_sonnet(monkeypatch):
    for var in _MODEL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    from generative import config

    importlib.reload(config)

    assert config.MODEL_MAIN == "anthropic/claude-sonnet-4-6"


def test_model_opus_alias_env_still_works(monkeypatch):
    """Bestehende .env-Dateien mit ATOMIC_AGENT_MODEL_OPUS duerfen nicht brechen."""
    monkeypatch.delenv("ATOMIC_AGENT_MODEL_MAIN", raising=False)
    monkeypatch.setenv("ATOMIC_AGENT_MODEL_OPUS", "anthropic/claude-opus-4-7")

    from generative import config

    importlib.reload(config)

    assert config.MODEL_MAIN == "anthropic/claude-opus-4-7"
    assert config.MODEL_OPUS == "anthropic/claude-opus-4-7"


def test_model_main_env_wins_over_opus_alias(monkeypatch):
    """Prioritaet: ATOMIC_AGENT_MODEL_MAIN > ATOMIC_AGENT_MODEL_OPUS > Default."""
    monkeypatch.setenv("ATOMIC_AGENT_MODEL_MAIN", "anthropic/claude-main-value")
    monkeypatch.setenv("ATOMIC_AGENT_MODEL_OPUS", "anthropic/claude-opus-value")

    from generative import config

    importlib.reload(config)

    assert config.MODEL_MAIN == "anthropic/claude-main-value"


def test_model_opus_is_deprecated_alias_of_model_main(monkeypatch):
    for var in _MODEL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    from generative import config

    importlib.reload(config)

    assert config.MODEL_OPUS == config.MODEL_MAIN


def test_model_opus_env_emits_deprecation_warning(monkeypatch):
    monkeypatch.delenv("ATOMIC_AGENT_MODEL_MAIN", raising=False)
    monkeypatch.setenv("ATOMIC_AGENT_MODEL_OPUS", "anthropic/claude-opus-4-7")

    from generative import config

    with pytest.warns(DeprecationWarning, match="ATOMIC_AGENT_MODEL_MAIN"):
        importlib.reload(config)


def test_model_judge_defaults_to_model_main(monkeypatch):
    monkeypatch.delenv("ATOMIC_AGENT_MODEL_JUDGE", raising=False)
    monkeypatch.setenv("ATOMIC_AGENT_MODEL_MAIN", "anthropic/claude-main-value")

    from generative import config

    importlib.reload(config)

    assert config.MODEL_JUDGE == "anthropic/claude-main-value"


def test_model_judge_env_override_wins(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_MODEL_MAIN", "anthropic/claude-main-value")
    monkeypatch.setenv("ATOMIC_AGENT_MODEL_JUDGE", "anthropic/claude-judge-value")

    from generative import config

    importlib.reload(config)

    assert config.MODEL_JUDGE == "anthropic/claude-judge-value"
    assert config.MODEL_MAIN == "anthropic/claude-main-value"


def test_role_env_override_is_independent_per_role(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_MODEL_PLANNER", "anthropic/claude-planner-value")

    from generative import config

    importlib.reload(config)

    assert config.MODEL_PLANNER == "anthropic/claude-planner-value"
    # andere Rollen bleiben beim Hauptslot-Default, unbeeinflusst vom Planner-Override
    assert config.MODEL_EXTRACTOR == config.MODEL_MAIN
    assert config.MODEL_CANONICALIZER == config.MODEL_MAIN


def test_role_constant_names_unchanged():
    """Sperrzone: planner.py/extractor.py/canonicalizer.py importieren diese Namen
    direkt — die Konstantennamen selbst duerfen sich durch #317 nicht aendern."""
    from generative import config

    assert hasattr(config, "MODEL_PLANNER")
    assert hasattr(config, "MODEL_EXTRACTOR")
    assert hasattr(config, "MODEL_EXTENDER")
    assert hasattr(config, "MODEL_CANONICALIZER")
