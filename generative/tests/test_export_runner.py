"""Tests für pipeline.export_runner — Verdrahtung von F1 (note_json) + F2
(portable_md) + F3 (export_convert) zu einer Format-Auswahl für CLI/GUI
(Output-Projekt F4). Reine Funktionslogik, kein argparse/FastAPI hier
(Orchestrator-/GUI-Wiring wird separat getestet, s. test_orchestrator_export.py
bzw. generative/gui/tests/test_app.py, test_runner.py, test_gui_settings.py).
"""

from __future__ import annotations

import json

import pytest

from generative.pipeline.export_runner import (
    EXPORT_FORMAT_CHOICES,
    FUTURE_FORMATS,
    parse_export_formats,
    requires_export_deps,
    run_export,
)
from generative.pipeline.note_json import dumps, note_to_json_dict, run_to_json_dict
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor
from generative.schemas.citation import CitationMeta


def _draft(**overrides) -> AtomicNoteDraft:
    base = dict(
        title="Zettelkasten-Methode",
        body="# Zettelkasten-Methode\n\nEin Zettelkasten ist ein Notizsystem (S. 4).",
        source_anchors=[TextAnchor(quote="Ein Zettelkasten ist ein Notizsystem", page="S. 4")],
        related=["[[Luhmann]]"],
        tags=["#konzept"],
        synthesis_confidence="high",
        quality_flags=[],
        critic_score=5,
        hard_gates_pass=True,
    )
    base.update(overrides)
    return AtomicNoteDraft(**base)


def _citation(**overrides) -> CitationMeta:
    base = dict(author="Luhmann, Niklas", year="1992", title="Zettelkasten", doi=None, source_file="luhmann.pdf")
    base.update(overrides)
    return CitationMeta(**base)


# --- parse_export_formats --------------------------------------------------


class TestParseExportFormats:
    def test_single_format(self):
        assert parse_export_formats("json") == ("json",)

    def test_order_preserved(self):
        assert parse_export_formats("pdf,docx,html") == ("pdf", "docx", "html")

    def test_dedup_order_preserving(self):
        assert parse_export_formats("json,pdf,json") == ("json", "pdf")

    def test_case_insensitive(self):
        assert parse_export_formats("JSON,Pdf") == ("json", "pdf")

    def test_trims_whitespace(self):
        assert parse_export_formats(" json , pdf ") == ("json", "pdf")

    def test_empty_string_returns_empty_tuple(self):
        assert parse_export_formats("") == ()

    def test_all_core_choices_valid(self):
        raw = ",".join(EXPORT_FORMAT_CHOICES)
        assert parse_export_formats(raw) == EXPORT_FORMAT_CHOICES

    def test_unknown_format_raises_with_valid_list(self):
        with pytest.raises(ValueError) as exc:
            parse_export_formats("bogus")
        msg = str(exc.value)
        assert "bogus" in msg
        for fmt in EXPORT_FORMAT_CHOICES:
            assert fmt in msg

    def test_future_format_raises_with_geplant_hint(self):
        assert "rtf" in FUTURE_FORMATS
        with pytest.raises(ValueError) as exc:
            parse_export_formats("rtf")
        msg = str(exc.value).lower()
        assert "geplant" in msg


class TestRequiresExportDeps:
    def test_pure_formats_need_no_deps(self):
        assert requires_export_deps(("json",)) is False
        assert requires_export_deps(("portable-md", "obsidian-md")) is False
        assert requires_export_deps(()) is False

    def test_binary_formats_need_deps(self):
        assert requires_export_deps(("pdf",)) is True
        assert requires_export_deps(("json", "docx")) is True
        for fmt in ("docx", "pdf", "html", "odt", "epub"):
            assert requires_export_deps((fmt,)) is True


# --- run_export: json --------------------------------------------------


class TestRunExportJson:
    def test_writes_per_note_and_gesamt_files(self, tmp_path):
        drafts = [_draft(title="Erste Note"), _draft(title="Zweite Note")]
        citation = _citation()
        written, messages = run_export(drafts, citation, ("json",), tmp_path)
        names = {p.name for p in written}
        assert "Erste Note.json" in names
        assert "Zweite Note.json" in names
        assert "luhmann-gesamt.json" in names
        assert messages == []

    def test_per_note_content_matches_note_to_json_dict(self, tmp_path):
        drafts = [_draft(title="Solo Note")]
        citation = _citation()
        written, _ = run_export(drafts, citation, ("json",), tmp_path, generated_at="2020-01-01")
        note_path = next(p for p in written if p.name == "Solo Note.json")
        expected = dumps(note_to_json_dict(drafts[0], citation, generated_at="2020-01-01"))
        assert note_path.read_text(encoding="utf-8") == expected
        # gültiges JSON, kein Zufalls-Fragment
        json.loads(note_path.read_text(encoding="utf-8"))

    def test_gesamt_content_matches_run_to_json_dict(self, tmp_path):
        drafts = [_draft(title="A"), _draft(title="B")]
        citation = _citation()
        written, _ = run_export(drafts, citation, ("json",), tmp_path, generated_at="2020-01-01")
        gesamt = next(p for p in written if p.name == "luhmann-gesamt.json")
        expected = dumps(run_to_json_dict(drafts, citation, generated_at="2020-01-01"))
        assert gesamt.read_text(encoding="utf-8") == expected


# --- run_export: portable-md --------------------------------------------


class TestRunExportPortableMd:
    def test_writes_per_note_and_gesamt_md(self, tmp_path):
        drafts = [_draft(title="Erste Note"), _draft(title="Zweite Note")]
        citation = _citation()
        written, messages = run_export(drafts, citation, ("portable-md",), tmp_path)
        names = {p.name for p in written}
        assert "Erste Note.md" in names
        assert "Zweite Note.md" in names
        assert "luhmann-gesamt.md" in names
        assert messages == []

    def test_internal_link_resolves_to_note_file_in_file_mode(self, tmp_path):
        # Note A verlinkt per Wikilink auf Note B — im file-mode muss der Link
        # auf "<slug-von-B>.md" zeigen (nicht auf einen Anchor, das ist run-mode).
        note_a = _draft(
            title="Note A",
            body="# Note A\n\nSiehe [[Note B]] für Details (S. 1).",
        )
        note_b = _draft(title="Note B", body="# Note B\n\nInhalt von B (S. 2).")
        citation = _citation()
        written, _ = run_export([note_a, note_b], citation, ("portable-md",), tmp_path)
        path_a = next(p for p in written if p.name == "Note A.md")
        text_a = path_a.read_text(encoding="utf-8")
        assert "(Note%20B.md)" in text_a or "(Note B.md)" in text_a


# --- run_export: binäre Formate (gemockt convert_portable_md) -----------


class TestRunExportBinaryFormats:
    def test_calls_convert_portable_md_with_expected_args(self, tmp_path, monkeypatch):
        calls = []

        def fake_convert(md_text, fmt, out_path, *, title=None, author=None, date=None, lang="de"):
            calls.append(
                {"fmt": fmt, "out_path": out_path, "title": title, "author": author, "date": date, "lang": lang}
            )
            out_path = out_path.__class__(out_path)
            out_path.write_bytes(b"stub")
            return out_path

        import generative.pipeline.export_convert as export_convert

        monkeypatch.setattr(export_convert, "export_available", lambda: (True, "stub ok"))
        monkeypatch.setattr(export_convert, "convert_portable_md", fake_convert)

        drafts = [_draft(title="Solo Note")]
        citation = _citation()
        written, messages = run_export(drafts, citation, ("docx",), tmp_path)

        assert messages == []
        note_calls = [c for c in calls if c["fmt"] == "docx" and c["out_path"].name == "Solo Note.docx"]
        assert len(note_calls) == 1
        assert note_calls[0]["title"] == "Solo Note"
        assert note_calls[0]["author"] == citation.author
        assert note_calls[0]["date"] == citation.display_year
        assert note_calls[0]["lang"] == "de"

        gesamt_calls = [c for c in calls if c["out_path"].name == "luhmann-gesamt.docx"]
        assert len(gesamt_calls) == 1

        names = {p.name for p in written}
        assert "Solo Note.docx" in names
        assert "luhmann-gesamt.docx" in names

    def test_missing_deps_raises_runtime_error_with_pip_hint(self, tmp_path, monkeypatch):
        import generative.pipeline.export_convert as export_convert

        monkeypatch.setattr(export_convert, "export_available", lambda: (False, "pandoc fehlt"))

        drafts = [_draft(title="Solo Note")]
        citation = _citation()
        with pytest.raises(RuntimeError) as exc:
            run_export(drafts, citation, ("docx",), tmp_path)
        msg = str(exc.value)
        assert "pandoc fehlt" in msg
        assert "pip install" in msg
        assert "atomic-notes[export]" in msg

    def test_json_and_portable_md_still_run_when_deps_missing(self, tmp_path, monkeypatch):
        # Formate VOR dem fehlgeschlagenen Binaerformat muessen trotzdem
        # geschrieben werden -- nur der Binaer-Teil bricht ab.
        import generative.pipeline.export_convert as export_convert

        monkeypatch.setattr(export_convert, "export_available", lambda: (False, "pandoc fehlt"))

        drafts = [_draft(title="Solo Note")]
        citation = _citation()
        with pytest.raises(RuntimeError):
            run_export(drafts, citation, ("json", "portable-md", "docx"), tmp_path)

        assert (tmp_path / "Solo Note.json").exists()
        assert (tmp_path / "Solo Note.md").exists()
        assert not (tmp_path / "Solo Note.docx").exists()


# --- run_export: obsidian-md ---------------------------------------------


class TestRunExportObsidianMd:
    def test_copies_written_files(self, tmp_path):
        src_dir = tmp_path / "vault-inbox"
        src_dir.mkdir()
        note_file = src_dir / "Erste Note.md"
        note_file.write_text("# Erste Note\n\nInhalt.", encoding="utf-8")
        export_root = tmp_path / "export"

        drafts = [_draft(title="Erste Note")]
        citation = _citation()
        written, messages = run_export(
            drafts, citation, ("obsidian-md",), export_root, written_files=[note_file], dry_run=False
        )
        copied = export_root / "Erste Note.md"
        assert copied.exists()
        assert copied.read_text(encoding="utf-8") == "# Erste Note\n\nInhalt."
        assert copied in written
        assert messages == []

    def test_dry_run_produces_skip_message_no_copy(self, tmp_path):
        export_root = tmp_path / "export"
        drafts = [_draft(title="Erste Note")]
        citation = _citation()
        written, messages = run_export(
            drafts, citation, ("obsidian-md",), export_root, written_files=[tmp_path / "irrelevant.md"], dry_run=True
        )
        assert written == []
        assert len(messages) == 1
        assert "obsidian-md" in messages[0]
        assert "dry" in messages[0].lower() or "vorschau" in messages[0].lower()

    def test_no_written_files_produces_distinct_skip_message(self, tmp_path):
        # Echter Lauf (dry_run=False) ohne written_files: die Meldung darf NICHT
        # faelschlich von einem Dry-Run sprechen (Review-Fund 3).
        export_root = tmp_path / "export"
        drafts = [_draft(title="Erste Note")]
        citation = _citation()
        written, messages = run_export(drafts, citation, ("obsidian-md",), export_root, written_files=None)
        assert written == []
        assert len(messages) == 1
        assert "obsidian-md" in messages[0]
        assert "dry" not in messages[0].lower()

    def test_missing_source_file_produces_visible_skip_message(self, tmp_path):
        # Review-Fund 2: eine nicht (mehr) existente Quelle darf nicht STILL
        # uebersprungen werden — sichtbare Meldung je Datei.
        src_dir = tmp_path / "vault-inbox"
        src_dir.mkdir()
        existing = src_dir / "Da.md"
        existing.write_text("# Da", encoding="utf-8")
        missing = src_dir / "Weg.md"  # nie geschrieben
        export_root = tmp_path / "export"

        drafts = [_draft(title="Da")]
        citation = _citation()
        written, messages = run_export(
            drafts, citation, ("obsidian-md",), export_root, written_files=[existing, missing], dry_run=False
        )
        assert (export_root / "Da.md").exists()
        assert len(messages) == 1
        assert "Weg.md" in messages[0]
        assert "obsidian-md" in messages[0]


# --- Titel-Kollision -------------------------------------------------------


class TestTitleCollision:
    def test_two_notes_same_title_get_suffix(self, tmp_path):
        drafts = [_draft(title="Doppelter Titel"), _draft(title="Doppelter Titel")]
        citation = _citation()
        written, _ = run_export(drafts, citation, ("json",), tmp_path)
        names = sorted(p.name for p in written if p.name != "luhmann-gesamt.json")
        assert names == ["Doppelter Titel-2.json", "Doppelter Titel.json"]
        # keine Datei wurde von der anderen ueberschrieben -- beide existieren
        # mit eigenem, unterscheidbarem Inhalt (unterschiedliche source_anchors
        # waeren noetig fuer echten Inhaltsvergleich; hier reicht Existenzpruefung).
        assert (tmp_path / "Doppelter Titel.json").exists()
        assert (tmp_path / "Doppelter Titel-2.json").exists()

    def test_note_titled_like_gesamt_file_does_not_clobber_it(self, tmp_path):
        # Review-Fund 4 (Mistral+Eigen): eine Note mit Titel exakt
        # "<pdf-stem>-gesamt" darf die Sammel-Datei nicht ueberschreiben --
        # der Sammel-Stem ist reserviert, die Note weicht auf -2 aus.
        drafts = [_draft(title="luhmann-gesamt")]
        citation = _citation()  # source_file=luhmann.pdf -> Sammel-Stem "luhmann-gesamt"
        written, _ = run_export(drafts, citation, ("json",), tmp_path, generated_at="2020-01-01")
        names = sorted(p.name for p in written)
        assert names == ["luhmann-gesamt-2.json", "luhmann-gesamt.json"]
        # Sammel-Datei enthaelt das run-Dict (notes-Liste), nicht die Einzel-Note
        gesamt = json.loads((tmp_path / "luhmann-gesamt.json").read_text(encoding="utf-8"))
        assert "notes" in gesamt
        note = json.loads((tmp_path / "luhmann-gesamt-2.json").read_text(encoding="utf-8"))
        assert note["note"]["title"] == "luhmann-gesamt"


class TestEmptyDrafts:
    def test_empty_drafts_skips_export_with_message(self, tmp_path):
        # Review-Fund 6: keine leeren -gesamt-Dateien fuer einen Lauf ohne Notes.
        written, messages = run_export([], _citation(), ("json", "portable-md"), tmp_path)
        assert written == []
        assert len(messages) == 1
        assert "keine Notes" in messages[0]
        assert list(tmp_path.iterdir()) == []  # export_root bleibt leer
