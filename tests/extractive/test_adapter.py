from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from extractive.pipeline.adapter import write_note
from shared.schemas.atomic_note_extractive import AtomicNoteExtractive


def _note(title: str) -> AtomicNoteExtractive:
    return AtomicNoteExtractive(
        title=title,
        concept_type="Concept",
        extracted_body=["This is a source-grounded sentence. (S. 1)"],
        source_anchors=[{"page": 1, "quote": "This is a source-grounded sentence.", "score": 1.0}],
        source_file="source.pdf",
        created="2026-05-26",
    )


@pytest.mark.parametrize(
    "title",
    [
        "../pwned",
        "..\\pwned",
        "C:\\absolute",
        "/tmp/outside",
        "foo:bar",
        "CON",
        "NUL",
        "bad\x00name",
    ],
)
def test_write_note_keeps_unsafe_titles_inside_out_dir(title: str):
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        path = write_note(_note(title), out_dir)

        assert path.parent == out_dir
        assert path.resolve().parent == out_dir.resolve()
        assert path.exists()


def test_write_note_preserves_normal_title():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        path = write_note(_note("Information Search Process"), out_dir)

        assert path == out_dir / "information-search-process.md"
        assert path.exists()
