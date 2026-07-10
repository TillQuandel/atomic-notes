"""Drift-Test: jedes AtomicNoteDraft-Feld braucht eine Ownership-Zeile (#99).

Erzwingt die Konvention aus dem Modul-Docstring von `atomic_note.py`: jedes
Feld trägt eine `# ownership: writer=<modul>[,...] reader=<modul>[,...]`-Zeile
direkt darüber oder inline auf der Feld-Zeile selbst. Ein neues Feld ohne
Ownership-Kommentar lässt diesen Test rot werden.

Quelltext-basiert (nicht nur `dataclasses.fields`): die Feldnamen kommen aus
der Reflection, der Ownership-Check selbst prüft den tatsächlichen
Kommentartext daneben — das ist nur über den Quelltext sichtbar.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

from generative.schemas.atomic_note import AtomicNoteDraft

_OWNERSHIP_RE = re.compile(r"#\s*ownership:\s*writer\s*=\s*\S.*reader\s*=\s*\S")


def _class_source_lines() -> list[str]:
    return inspect.getsource(AtomicNoteDraft).splitlines()


def _field_def_line(lines: list[str], field_name: str) -> int:
    """Zeilen-Index der Feld-Definition (erste Zeile — bei Multi-Line-Defaults
    wie `hub_subconcept_descriptions` zählt die Startzeile). Anker `^\\s*name\\s*:`
    ist präfix-sicher: kein Feldname von AtomicNoteDraft ist Präfix eines
    anderen (hub_subconcepts/hub_subconcept_descriptions unterscheiden sich
    schon am nächsten Zeichen nach dem gemeinsamen Stamm)."""
    pattern = re.compile(rf"^\s*{re.escape(field_name)}\s*:")
    for i, line in enumerate(lines):
        if pattern.match(line):
            return i
    raise AssertionError(f"Feld '{field_name}' nicht im Quelltext von AtomicNoteDraft gefunden (Klasse umgebaut?)")


def test_all_atomic_note_draft_fields_have_ownership_comment() -> None:
    lines = _class_source_lines()
    offenders: list[str] = []
    for f in dataclasses.fields(AtomicNoteDraft):
        idx = _field_def_line(lines, f.name)
        inline_ok = bool(_OWNERSHIP_RE.search(lines[idx]))
        above_ok = idx > 0 and bool(_OWNERSHIP_RE.search(lines[idx - 1]))
        if not (inline_ok or above_ok):
            offenders.append(f.name)

    assert not offenders, (
        "Feld(er) ohne ownership-Kommentar — siehe #99-Konvention im Modul-Docstring "
        "von generative/schemas/atomic_note.py: " + ", ".join(offenders)
    )
