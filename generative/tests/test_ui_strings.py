"""Tests für den zweisprachigen Runtime-UX-String-Katalog (generative.ui_strings, #157)."""

from __future__ import annotations

import pytest

from generative import ui_strings
from generative.ui_strings import STRINGS, lang, msg


def test_default_language_is_english(monkeypatch):
    monkeypatch.delenv("ATOMIC_AGENT_UI_LANGUAGE", raising=False)
    assert lang() == "en"
    assert msg("cli.unknown_command", cmd="x") == "Unknown command: x"


def test_de_switch(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_UI_LANGUAGE", "de")
    assert lang() == "de"
    assert msg("cli.unknown_command", cmd="x") == "Unbekanntes Kommando: x"


def test_unknown_language_falls_back_to_english(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_UI_LANGUAGE", "fr")
    assert lang() == "en"
    assert msg("cli.unknown_command", cmd="x") == "Unknown command: x"


def test_empty_language_falls_back_to_english(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_UI_LANGUAGE", "")
    assert lang() == "en"


def test_language_value_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_UI_LANGUAGE", "DE")
    assert lang() == "de"


def test_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        msg("does.not.exist")


def test_every_key_has_en_and_de():
    """Vollständigkeit: kein halb-übersetzter Katalog (jeder Key hat en UND de)."""
    incomplete = {k: sorted(v) for k, v in STRINGS.items() if set(v) != {"en", "de"}}
    assert incomplete == {}, f"Keys ohne beide Sprachen: {incomplete}"


def test_all_values_nonempty():
    empty = [f"{k}.{sub}" for k, v in STRINGS.items() for sub, s in v.items() if not s.strip()]
    assert empty == [], f"Leere Katalog-Werte: {empty}"


def test_msg_formats_placeholders():
    assert "42" in msg("cli.port_out_of_range", port=42)


def test_doctor_pointer_is_language_invariant():
    # Der doctor-Verweis ist ein Kommando-Name, kein zu übersetzender Text.
    assert ui_strings.DOCTOR_POINTER == "→ atomic-notes doctor"
