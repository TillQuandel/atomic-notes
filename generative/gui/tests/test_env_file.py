"""Tests fuer den .env-Merge-Write (B1b): reine Datei-/Validierungslogik,
kein FastAPI (analog test_gui_settings.py/test_run_history.py).

WICHTIG: alle Tests schreiben ausschliesslich in `tmp_path` -- niemals in die
echte `generative/.env`.
"""

from __future__ import annotations

from generative.gui import env_file


# --- validate_key_value --------------------------------------------------


def test_validate_key_value_accepts_normal_key():
    value, error = env_file.validate_key_value("sk-test-xxx")
    assert error is None
    assert value == "sk-test-xxx"


def test_validate_key_value_trims_surrounding_whitespace():
    value, error = env_file.validate_key_value("  sk-test-xxx  ")
    assert error is None
    assert value == "sk-test-xxx"


def test_validate_key_value_rejects_none():
    value, error = env_file.validate_key_value(None)
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_non_string():
    value, error = env_file.validate_key_value(12345)
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_empty_string():
    value, error = env_file.validate_key_value("")
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_whitespace_only():
    value, error = env_file.validate_key_value("   ")
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_newline():
    value, error = env_file.validate_key_value("sk-x\nATOMIC_AGENT_BACKEND=evil")
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_carriage_return():
    value, error = env_file.validate_key_value("sk-x\rsecond-line")
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_tab():
    value, error = env_file.validate_key_value("sk-x\ttab")
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_nul():
    value, error = env_file.validate_key_value("sk-x\x00nul")
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_del():
    value, error = env_file.validate_key_value("sk-x\x7fdel")
    assert value is None
    assert error is not None


def test_validate_key_value_rejects_trailing_newline_even_though_strip_would_remove_it():
    # `.strip()` wuerde ein reines Trailing-`\n` entfernen -- die Pruefung
    # muss trotzdem VOR dem Trimmen greifen (sonst waere ein Trailing-`\n`
    # unsichtbar erlaubt statt abgelehnt).
    value, error = env_file.validate_key_value("sk-test-xxx\n")
    assert value is None
    assert error is not None


# --- write_env_var ---------------------------------------------------------


def test_write_env_var_creates_file_if_missing(tmp_path):
    path = tmp_path / ".env"
    assert not path.exists()
    env_file.write_env_var("ANTHROPIC_API_KEY", "sk-test-xxx", path)
    assert path.read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=sk-test-xxx\n"


def test_write_env_var_appends_when_var_not_present(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# Kommentar\nOTHER_VAR=1\n", encoding="utf-8")
    env_file.write_env_var("OPENAI_API_KEY", "sk-openai-xxx", path)
    content = path.read_text(encoding="utf-8")
    assert content == "# Kommentar\nOTHER_VAR=1\nOPENAI_API_KEY=sk-openai-xxx\n"


def test_write_env_var_merge_roundtrip(tmp_path):
    path = tmp_path / ".env"
    original = (
        "# Kommentar oben\n"
        "\n"
        "ATOMIC_AGENT_VAULT_PATH=/some/vault\n"
        "ANTHROPIC_API_KEY=alt\n"
        "ATOMIC_AGENT_BACKEND=subscription\n"
    )
    path.write_text(original, encoding="utf-8")
    env_file.write_env_var("ANTHROPIC_API_KEY", "neu", path)
    content = path.read_text(encoding="utf-8")
    assert content == (
        "# Kommentar oben\n"
        "\n"
        "ATOMIC_AGENT_VAULT_PATH=/some/vault\n"
        "ANTHROPIC_API_KEY=neu\n"
        "ATOMIC_AGENT_BACKEND=subscription\n"
    )


def test_write_env_var_replaces_only_first_occurrence(tmp_path):
    path = tmp_path / ".env"
    path.write_text("ANTHROPIC_API_KEY=alt1\nANTHROPIC_API_KEY=alt2\n", encoding="utf-8")
    env_file.write_env_var("ANTHROPIC_API_KEY", "neu", path)
    content = path.read_text(encoding="utf-8")
    assert content == "ANTHROPIC_API_KEY=neu\nANTHROPIC_API_KEY=alt2\n"


def test_write_env_var_does_not_touch_similarly_prefixed_var(tmp_path):
    # "ANTHROPIC_API_KEY_OLD=..." darf NICHT durch ein Update von
    # "ANTHROPIC_API_KEY" getroffen werden (Praefix-Kollision).
    path = tmp_path / ".env"
    path.write_text("ANTHROPIC_API_KEY_OLD=bleibt\n", encoding="utf-8")
    env_file.write_env_var("ANTHROPIC_API_KEY", "neu", path)
    content = path.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY_OLD=bleibt" in content
    assert "ANTHROPIC_API_KEY=neu" in content


def test_write_env_var_adds_missing_trailing_newline_before_append(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OTHER_VAR=1", encoding="utf-8")  # kein Newline am Ende
    env_file.write_env_var("OPENAI_API_KEY", "sk-openai-xxx", path)
    content = path.read_text(encoding="utf-8")
    assert content == "OTHER_VAR=1\nOPENAI_API_KEY=sk-openai-xxx\n"


def test_write_env_var_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / ".env"
    env_file.write_env_var("ANTHROPIC_API_KEY", "sk-test-xxx", path)
    assert path.exists()
