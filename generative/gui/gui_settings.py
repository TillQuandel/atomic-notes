"""Persistierte GUI-Einstellungen (P2): zuletzt gewaehlte Lauf-Optionen
ueberleben einen GUI-Neustart.

Bewusst NICHT das `settings.json` aus der Atomic-Agent-Provider-Abstraktion
(Pipeline-Config) -- eine GUI-eigene Datei unter `generative/.cache/gui/`
(Default-Pfad in app.py, `_DEFAULT_SETTINGS_PATH`).

Reine Datei-/Validierungslogik, kein FastAPI (analog run_history.py). Die
Backend-/Profil-Wertepruefung wird von `app._validate_run_options`
wiederverwendet (`validate_backend`/`validate_profile`) statt dupliziert.

Schema: `{backend?, profile?, no_llm?, dry_run?}` -- alle vier Keys optional,
nur tatsaechlich gesetzte Werte werden gespeichert/gelesen. Keine Secrets
(API-Keys bleiben in `.env`).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from generative.runtime_config import PRESETS

logger = logging.getLogger(__name__)

BACKENDS = frozenset({"subscription", "litellm"})
SETTINGS_KEYS = frozenset({"backend", "profile", "no_llm", "dry_run"})


def validate_backend(value) -> tuple[str | None, str | None]:
    """Einzelnen Backend-Wert pruefen. Leer/None = kein Override (Server-Default)."""
    if value in (None, ""):
        return None, None
    if value not in BACKENDS:
        return None, f"Unbekannter Backend-Wert: {value!r} (erlaubt: {', '.join(sorted(BACKENDS))})"
    return value, None


def validate_profile(value) -> tuple[str | None, str | None]:
    """Einzelnen Profil-Wert pruefen. Leer/None = kein Override (Server-Default)."""
    if value in (None, ""):
        return None, None
    if value not in PRESETS:
        return None, f"Unbekanntes Profil: {value!r} (erlaubt: {', '.join(sorted(PRESETS))})"
    return value, None


def _validate_bool(value, field_name: str) -> tuple[bool | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, bool):
        return None, f"{field_name} muss ein Boolean sein."
    return value, None


def validate_settings(payload) -> tuple[dict, str | None]:
    """Validiert+normalisiert ein `PUT /api/settings`-Payload.

    Rueckgabe: (normalisiert, fehlermeldung). Anders als bei
    `app._validate_run_options` (dort heisst `no_llm=False` "keine Option
    gesetzt" und wird weggelassen): hier ist `False` ein bewusst gesetzter
    Wert -- z.B. `dry_run=False` bedeutet "Standard ist Schreib-Modus" -- und
    bleibt im Ergebnis erhalten (sonst Rundreise-Bruch nach GUI-Neustart).
    """
    if not isinstance(payload, dict):
        return {}, "Payload muss ein Objekt sein."
    unknown = set(payload) - SETTINGS_KEYS
    if unknown:
        return {}, f"Unbekannte Einstellung(en): {', '.join(sorted(unknown))}"

    normalized: dict = {}

    backend, error = validate_backend(payload.get("backend"))
    if error:
        return {}, error
    if backend is not None:
        normalized["backend"] = backend

    profile, error = validate_profile(payload.get("profile"))
    if error:
        return {}, error
    if profile is not None:
        normalized["profile"] = profile

    no_llm, error = _validate_bool(payload.get("no_llm"), "no_llm")
    if error:
        return {}, error
    if no_llm is not None:
        normalized["no_llm"] = no_llm

    dry_run, error = _validate_bool(payload.get("dry_run"), "dry_run")
    if error:
        return {}, error
    if dry_run is not None:
        normalized["dry_run"] = dry_run

    return normalized, None


def write_settings(settings: dict, path: Path | str) -> Path:
    """Schreibt die Settings atomar (Tempfile im selben Verzeichnis + `os.replace`),
    exakt wie `run_history.write_run_record`. Ersetzt die Datei vollstaendig --
    kein Merge mit vorherigem Inhalt (der Aufrufer schickt bereits das komplette
    Objekt, s. Plan P2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def read_settings(path: Path | str) -> tuple[dict, str | None]:
    """Liest die Settings. Fehlende Datei -> `({}, None)` (kein Fehler -- noch
    nie gespeichert). Kaputte/nicht lesbare Datei -> `({}, warnung)` (fail-open
    lesend, L5: sichtbar statt still). Unbekannte Keys in der Datei werden
    stillschweigend gefiltert (z.B. Reste eines frueheren Schemas)."""
    path = Path(path)
    if not path.is_file():
        return {}, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Kaputte GUI-Settings-Datei uebersprungen (%s): %s", path, exc)
        return (
            {},
            "Gespeicherte Einstellungen konnten nicht gelesen werden (Datei beschädigt) — Standardwerte werden verwendet.",
        )
    if not isinstance(data, dict):
        logger.warning("GUI-Settings-Datei kein Objekt, ignoriert: %s", path)
        return {}, "Gespeicherte Einstellungen waren ungültig — Standardwerte werden verwendet."
    return {k: v for k, v in data.items() if k in SETTINGS_KEYS}, None
