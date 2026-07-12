"""Direkte Unit-Tests fuer shared/path_safety.py (Traversal-Schutz).

Migriert aus tests/extractive/test_adapter.py: dort wurde der Schutz nur
indirekt ueber extractive.pipeline.adapter.write_note geprueft, was jinja2 +
pydantic (das [extractive]-Extra) verlangt und deshalb aus der kanonischen
CI-Suite fiel. path_safety selbst nutzt nur die stdlib — hier direkt und
extra-frei getestet, damit das sicherheitsrelevante Shared-Modul CI-abgedeckt ist.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from shared.path_safety import contained_child_path, resolve_source_path, safe_filename_stem

# Unsichere Titel aus dem alten write_note-Test: Traversal, absolute Pfade,
# Windows-Sondernamen, NUL-Byte. Der Slug muss jeden davon im Zielordner halten.
UNSAFE_TITLES = [
    "../pwned",
    "..\\pwned",
    "C:\\absolute",
    "/tmp/outside",
    "foo:bar",
    "CON",
    "NUL",
    "bad\x00name",
]


@pytest.mark.parametrize("title", UNSAFE_TITLES)
def test_safe_filename_stem_neutralises_unsafe_titles(title: str):
    stem = safe_filename_stem(title)
    assert stem, "Slug darf nie leer sein"
    assert "/" not in stem
    assert "\\" not in stem
    assert ".." not in stem
    assert stem.upper() not in {"CON", "PRN", "AUX", "NUL"}


@pytest.mark.parametrize("title", UNSAFE_TITLES)
def test_unsafe_title_stays_inside_out_dir(title: str):
    """End-to-end von path_safety: Slug + Containment-Check bleiben im Zielordner."""
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        stem = safe_filename_stem(title)
        path = contained_child_path(out_dir, f"{stem}.md")
        assert path.parent == out_dir
        assert path.resolve().parent == out_dir.resolve()


def test_safe_filename_stem_preserves_normal_title():
    assert safe_filename_stem("Information Search Process") == "information-search-process"


def test_safe_filename_stem_reserved_name_gets_fallback():
    assert safe_filename_stem("CON") == "con-note"


def test_safe_filename_stem_empty_falls_back():
    assert safe_filename_stem("") == "note"
    assert safe_filename_stem("   ") == "note"
    assert safe_filename_stem("...") == "note"
    assert safe_filename_stem("", fallback="x") == "x"


def test_safe_filename_stem_truncates_to_max_len():
    assert len(safe_filename_stem("a" * 200)) <= 60
    assert len(safe_filename_stem("a" * 200, max_len=10)) <= 10


def test_contained_child_path_accepts_normal_child():
    with tempfile.TemporaryDirectory() as d:
        parent = Path(d)
        child = contained_child_path(parent, "note.md")
        assert child == parent / "note.md"


@pytest.mark.parametrize(
    "evil",
    ["../escape.md", "../../escape.md", f"..{os.sep}escape.md", f"sub{os.sep}..{os.sep}..{os.sep}escape.md"],
)
def test_contained_child_path_rejects_traversal(evil: str):
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            contained_child_path(Path(d), evil)


def test_contained_child_path_rejects_absolute_sibling():
    with tempfile.TemporaryDirectory() as inside, tempfile.TemporaryDirectory() as outside:
        target = os.path.join(outside, "evil.md")
        with pytest.raises(ValueError):
            contained_child_path(Path(inside), target)


# -- resolve_source_path: Apostroph-/Anfuehrungszeichen-Varianten (#186) --------


def test_resolve_source_path_returns_exact_match_unchanged(tmp_path: Path):
    f = tmp_path / "plain.pdf"
    f.write_text("x")
    assert resolve_source_path(f) == f


def test_resolve_source_path_finds_curly_apostrophe_file_via_straight_query(tmp_path: Path):
    """Datei liegt mit typografischem Apostroph (U+2019) vor, Nutzer tippt den geraden ' -- muss trotzdem gefunden werden."""
    f = tmp_path / "Porst’s-Buch.pdf"
    f.write_text("x")
    queried = tmp_path / "Porst's-Buch.pdf"
    assert resolve_source_path(queried) == f


def test_resolve_source_path_finds_straight_apostrophe_file_via_curly_query(tmp_path: Path):
    """Umgekehrter Fall: Datei mit geradem ', Anfrage mit typografischem U+2019."""
    f = tmp_path / "Porst's-Buch.pdf"
    f.write_text("x")
    queried = tmp_path / "Porst’s-Buch.pdf"
    assert resolve_source_path(queried) == f


def test_resolve_source_path_raises_with_repr_when_no_candidate(tmp_path: Path):
    missing = tmp_path / "nichts-hier.pdf"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_source_path(missing)
    assert repr(str(missing)) in str(exc_info.value)


def test_resolve_source_path_raises_and_lists_candidates_when_ambiguous(tmp_path: Path):
    a = tmp_path / "Porst’s-Buch.pdf"
    b = tmp_path / "Porst‘s-Buch.pdf"
    a.write_text("x")
    b.write_text("x")
    queried = tmp_path / "Porst's-Buch.pdf"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_source_path(queried)
    msg = str(exc_info.value)
    assert repr(str(queried)) in msg
    assert str(a) in msg
    assert str(b) in msg


def test_resolve_source_path_no_fallback_without_quote_chars(tmp_path: Path):
    """Fehlt ein Apostroph-Zeichen ganz (Tippfehler ohne Zweifelszeichen), soll kein Glob-Fallback greifen."""
    missing = tmp_path / "voellig-anderer-name.pdf"
    (tmp_path / "anderes-dokument.pdf").write_text("x")
    with pytest.raises(FileNotFoundError):
        resolve_source_path(missing)


# -- Nachbesserung (#186 Review): das `?`-Glob matcht JEDES Zeichen, nicht nur
# Quote-Varianten -- ohne strikten Re-Filter wuerden beliebige Ein-Zeichen-
# Abweichungen (Buchstabe/Ziffer/Unterstrich) faelschlich als "Apostroph-
# Variante" durchgehen. -----------------------------------------------------


@pytest.mark.parametrize(
    "wrong_name",
    ["PorstXs-Buch.pdf", "Porst_s-Buch.pdf", "Porst5s-Buch.pdf"],
)
def test_resolve_source_path_rejects_non_quote_char_glob_matches(tmp_path: Path, wrong_name: str):
    """`?` im Glob-Pattern matcht auch Nicht-Quote-Zeichen -- das darf NICHT als
    Apostroph-Variante durchgehen (Fehlmatch aus Cross-Model-Review, 3x konvergent)."""
    (tmp_path / wrong_name).write_text("x")
    queried = tmp_path / "Porst's-Buch.pdf"
    with pytest.raises(FileNotFoundError):
        resolve_source_path(queried)


def test_resolve_source_path_ignores_directory_candidates(tmp_path: Path):
    """Ein gleichnamiges VERZEICHNIS (Apostroph-Variante im Namen) darf nie als
    Datei-Fallback zurueckgegeben werden."""
    (tmp_path / "Porst’s-Buch.pdf").mkdir()
    queried = tmp_path / "Porst's-Buch.pdf"
    with pytest.raises(FileNotFoundError):
        resolve_source_path(queried)
