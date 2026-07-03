"""Tests fuer die persistierten GUI-Einstellungen (P2): reine Datei-/
Validierungslogik, kein FastAPI (analog test_run_history.py).

Schema (Plan P2 + Task-Kontext-Update): {backend, profile, no_llm, dry_run},
jeweils optional -- nur tatsaechlich gesetzte Keys werden gespeichert/gelesen.
"""

from __future__ import annotations

import json

from generative.gui import gui_settings


# --- validate_settings -------------------------------------------------


def test_validate_settings_empty_payload_ok():
    normalized, error = gui_settings.validate_settings({})
    assert error is None
    assert normalized == {}


def test_validate_settings_accepts_all_four_keys():
    normalized, error = gui_settings.validate_settings(
        {"backend": "litellm", "profile": "fast", "no_llm": True, "dry_run": False}
    )
    assert error is None
    assert normalized == {"backend": "litellm", "profile": "fast", "no_llm": True, "dry_run": False}


def test_validate_settings_keeps_explicit_false_for_no_llm_and_dry_run():
    # Anders als bei POST /api/run-options: False ist hier ein bewusst gesetzter
    # Wert (z.B. "Standard ist Schreib-Modus") und darf nicht verschwinden.
    normalized, error = gui_settings.validate_settings({"no_llm": False, "dry_run": False})
    assert error is None
    assert normalized == {"no_llm": False, "dry_run": False}


def test_validate_settings_empty_string_backend_and_profile_means_unset():
    normalized, error = gui_settings.validate_settings({"backend": "", "profile": ""})
    assert error is None
    assert normalized == {}


def test_validate_settings_rejects_unknown_key():
    normalized, error = gui_settings.validate_settings({"foo": "bar"})
    assert normalized == {}
    assert "foo" in error


def test_validate_settings_rejects_unknown_backend_value():
    normalized, error = gui_settings.validate_settings({"backend": "openai-direct"})
    assert normalized == {}
    assert "backend" in error.lower()


def test_validate_settings_rejects_unknown_profile_value():
    normalized, error = gui_settings.validate_settings({"profile": "turbo"})
    assert normalized == {}
    assert "profil" in error.lower()


def test_validate_settings_rejects_non_bool_no_llm():
    normalized, error = gui_settings.validate_settings({"no_llm": "yes"})
    assert normalized == {}
    assert error is not None


def test_validate_settings_rejects_non_bool_dry_run():
    normalized, error = gui_settings.validate_settings({"dry_run": "yes"})
    assert normalized == {}
    assert error is not None


def test_validate_settings_rejects_non_dict_payload():
    normalized, error = gui_settings.validate_settings(["not", "a", "dict"])
    assert normalized == {}
    assert error is not None


# --- vault_path (B2: Vault-/Ordner-Wahl) --------------------------------


def test_validate_settings_accepts_vault_path_string():
    normalized, error = gui_settings.validate_settings({"vault_path": "C:/Users/x/Vault"})
    assert error is None
    assert normalized == {"vault_path": "C:/Users/x/Vault"}


def test_validate_settings_rejects_empty_vault_path():
    # Anders als backend/profile (dort bedeutet "" "Server-Default"): ein leerer
    # vault_path hat keine sinnvolle Bedeutung -- Fehler statt stillem Weglassen.
    normalized, error = gui_settings.validate_settings({"vault_path": ""})
    assert normalized == {}
    assert error is not None


def test_validate_settings_rejects_non_string_vault_path():
    normalized, error = gui_settings.validate_settings({"vault_path": 123})
    assert normalized == {}
    assert error is not None


def test_validate_settings_vault_path_no_existence_check():
    # validate_settings prueft NICHT ob der Pfad existiert -- das macht der
    # Endpunkt (PUT /api/vault). Ein beliebiger nicht-leerer String ist hier gueltig.
    normalized, error = gui_settings.validate_settings({"vault_path": "/nicht/vorhanden"})
    assert error is None
    assert normalized == {"vault_path": "/nicht/vorhanden"}


# --- write_settings / read_settings ------------------------------------


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "gui" / "settings.json"
    gui_settings.write_settings({"backend": "litellm", "dry_run": False}, path)
    data, warning = gui_settings.read_settings(path)
    assert warning is None
    assert data == {"backend": "litellm", "dry_run": False}


def test_read_missing_file_returns_empty_no_warning(tmp_path):
    data, warning = gui_settings.read_settings(tmp_path / "nope" / "settings.json")
    assert data == {}
    assert warning is None


def test_read_corrupt_json_returns_empty_with_warning(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json", encoding="utf-8")
    data, warning = gui_settings.read_settings(path)
    assert data == {}
    assert warning is not None


def test_read_non_object_json_returns_empty_with_warning(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(["a", "list"]), encoding="utf-8")
    data, warning = gui_settings.read_settings(path)
    assert data == {}
    assert warning is not None


def test_read_filters_unknown_keys_from_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"backend": "litellm", "api_key": "secret"}), encoding="utf-8")
    data, warning = gui_settings.read_settings(path)
    assert warning is None
    assert data == {"backend": "litellm"}
    assert "api_key" not in data


def test_write_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "settings.json"
    gui_settings.write_settings({"profile": "fast"}, path)
    assert path.exists()


def test_write_then_read_roundtrip_with_vault_path(tmp_path):
    path = tmp_path / "gui" / "settings.json"
    gui_settings.write_settings({"backend": "litellm", "vault_path": "C:/Vault"}, path)
    data, warning = gui_settings.read_settings(path)
    assert warning is None
    assert data == {"backend": "litellm", "vault_path": "C:/Vault"}
