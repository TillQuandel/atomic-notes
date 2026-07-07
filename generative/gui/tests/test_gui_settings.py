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


# --- vault_path SSoT (S4, #150): nicht ueber validate_settings setzbar -----


def test_validate_settings_ignores_vault_path_string():
    # S4 (#150): `vault_path` ist NICHT ueber PUT /api/settings setzbar -- er wird
    # ignoriert (kein Fehler, damit ein alter Client keinen 422 kassiert) und
    # taucht NICHT im normalisierten Ergebnis auf. Nur PUT /api/vault aendert ihn.
    normalized, error = gui_settings.validate_settings({"vault_path": "C:/Users/x/Vault"})
    assert error is None
    assert "vault_path" not in normalized


def test_validate_settings_ignores_vault_path_alongside_valid_keys():
    # Andere Keys bleiben gueltig, nur vault_path faellt weg.
    normalized, error = gui_settings.validate_settings({"backend": "litellm", "vault_path": "C:/x"})
    assert error is None
    assert normalized == {"backend": "litellm"}


def test_validate_settings_ignores_empty_or_nonstring_vault_path():
    # Auch ungueltige vault_path-Werte fuehren nicht mehr zu einem Fehler --
    # sie werden schlicht ignoriert (der Endpunkt bewahrt den persistierten Wert).
    for bad in ("", 123, None):
        normalized, error = gui_settings.validate_settings({"vault_path": bad})
        assert error is None, bad
        assert "vault_path" not in normalized


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


# --- export_formats (F4: Export-Format-Auswahl als Lauf-Option) -----------


def test_validate_export_formats_none_means_unset():
    value, error = gui_settings.validate_export_formats(None)
    assert error is None
    assert value is None


def test_validate_export_formats_empty_list_ok():
    value, error = gui_settings.validate_export_formats([])
    assert error is None
    assert value == []


def test_validate_export_formats_accepts_core_formats():
    value, error = gui_settings.validate_export_formats(["portable-md", "docx", "pdf", "html", "json"])
    assert error is None
    assert value == ["portable-md", "docx", "pdf", "html", "json"]


def test_validate_export_formats_accepts_odt_epub():
    value, error = gui_settings.validate_export_formats(["odt", "epub"])
    assert error is None
    assert value == ["odt", "epub"]


def test_validate_export_formats_rejects_obsidian_md():
    # obsidian-md gibt es in der GUI ohnehin als Download (keine GUI-Format-Option).
    value, error = gui_settings.validate_export_formats(["obsidian-md"])
    assert value is None
    assert error is not None
    assert "obsidian-md" in error


def test_validate_export_formats_rejects_unknown_format():
    value, error = gui_settings.validate_export_formats(["bogus"])
    assert value is None
    assert error is not None
    assert "bogus" in error


def test_validate_export_formats_rejects_non_list():
    value, error = gui_settings.validate_export_formats("docx")
    assert value is None
    assert error is not None


def test_validate_export_formats_rejects_non_string_items():
    value, error = gui_settings.validate_export_formats(["docx", 1])
    assert value is None
    assert error is not None


def test_validate_settings_accepts_export_formats():
    normalized, error = gui_settings.validate_settings({"export_formats": ["docx", "pdf"]})
    assert error is None
    assert normalized == {"export_formats": ["docx", "pdf"]}


def test_validate_settings_accepts_empty_export_formats_list():
    normalized, error = gui_settings.validate_settings({"export_formats": []})
    assert error is None
    assert normalized == {"export_formats": []}


def test_validate_settings_rejects_invalid_export_formats():
    normalized, error = gui_settings.validate_settings({"export_formats": ["bogus"]})
    assert normalized == {}
    assert error is not None


def test_write_then_read_roundtrip_with_export_formats(tmp_path):
    path = tmp_path / "gui" / "settings.json"
    gui_settings.write_settings({"export_formats": ["docx", "html"]}, path)
    data, warning = gui_settings.read_settings(path)
    assert warning is None
    assert data == {"export_formats": ["docx", "html"]}


def test_validate_export_formats_normalizes_case(tmp_path):
    # Review-Fund 5 (Mistral): konsistent zur CLI (parse_export_formats ist
    # case-insensitiv) -- "JSON" wird akzeptiert und kanonisch (lowercase) gespeichert.
    value, error = gui_settings.validate_export_formats(["JSON", "Pdf"])
    assert error is None
    assert value == ["json", "pdf"]


def test_validate_export_formats_strips_whitespace():
    value, error = gui_settings.validate_export_formats([" docx "])
    assert error is None
    assert value == ["docx"]


def test_validate_export_formats_dedupes_order_preserving():
    value, error = gui_settings.validate_export_formats(["json", "pdf", "JSON"])
    assert error is None
    assert value == ["json", "pdf"]
