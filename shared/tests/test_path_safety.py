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

from shared.path_safety import contained_child_path, safe_filename_stem

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
