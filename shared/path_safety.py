from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_DASH_RUN = re.compile(r"-+")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Apostroph-/Anfuehrungszeichen-Varianten, die beim Kopieren aus PDFs/Web
# unbemerkt gegeneinander vertauscht werden (#186): gerades Apostroph, rechtes/
# linkes typografisches Apostroph, Gravis, Akut.
_QUOTE_LIKE_CHARS = "'’‘`´"


def safe_filename_stem(value: str, *, max_len: int = 60, fallback: str = "note") -> str:
    stem = str(value or "").lower().replace(" ", "-")
    stem = _UNSAFE_FILENAME_CHARS.sub("-", stem)
    stem = _DASH_RUN.sub("-", stem).strip(" .-_")
    if not stem:
        stem = fallback
    stem = stem[:max_len].strip(" .-_") or fallback
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"{stem}-{fallback}"
    return stem


def resolve_source_path(source: str | Path) -> Path:
    """Loest einen Quell-Dateipfad (z. B. PDF) auf, inkl. Glob-Fallback fuer
    Apostroph-/Anfuehrungszeichen-Varianten (#186).

    Existiert `source` exakt, wird er unveraendert zurueckgegeben. Sonst wird
    im selben Verzeichnis nach Dateien gesucht, deren Name bis auf gerade vs.
    typografische Apostrophe/Anfuehrungszeichen (' ’ ‘ ` ´)
    identisch ist. Genau ein Treffer wird verwendet (Hinweis auf stderr);
    bei 0 oder >1 Treffern wirft die Funktion FileNotFoundError mit
    `repr(str(source))` und -- falls mehrdeutig -- den Kandidaten, damit alle
    Aufrufer dieselbe, gut lesbare Fehlermeldung erhalten (statt sie je Stelle
    zu duplizieren).
    """
    path = Path(source)
    if path.exists():
        return path

    pattern = glob.escape(path.name)
    for ch in _QUOTE_LIKE_CHARS:
        pattern = pattern.replace(ch, "?")

    candidates = sorted(path.parent.glob(pattern)) if path.parent.is_dir() else []

    if len(candidates) == 1:
        print(
            f"Hinweis: {repr(str(path))} nicht gefunden, verwende Fallback-Treffer "
            f"(Apostroph-/Anfuehrungszeichen-Variante): {candidates[0]}",
            file=sys.stderr,
        )
        return candidates[0]

    if candidates:
        listing = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"{repr(str(path))} nicht eindeutig -- mehrere Kandidaten gefunden: {listing}")

    raise FileNotFoundError(repr(str(path)))


def contained_child_path(parent: Path, filename: str) -> Path:
    candidate = parent / filename
    base = os.path.abspath(os.path.normpath(os.fspath(parent)))
    target = os.path.abspath(os.path.normpath(os.path.join(base, filename)))
    if os.path.commonpath([base, target]) != base:
        raise ValueError(f"unsafe output path outside target directory: {filename}")
    return candidate
