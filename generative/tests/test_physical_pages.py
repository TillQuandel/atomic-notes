"""Tests für Issue #95: Seitenangaben aus PDFs ohne `/PageLabels` als PDF-Position
kennzeichnen statt sie unmarkiert als gedruckte Seite auszugeben.

Signal-Fluss: `pdf_chunker.pdf_uses_physical_pages()` (Zweit-Check auf
`_pdf_page_labels`, analog zum bestehenden Edition-Verifikations-Check in
`orchestrator.main()`, Zeile ~2244) -> `CitationMeta.physical_pages` (additives
Feld, #96-E3a-Nachzug) -> Render-Schicht (`convert_inline_to_footnotes`/
`build_quellen_block`/`portable_md`) hängt „PDF-" vor das `S.`-Label.

Das Pipeline-INTERNE `(S. N)`-Inline-Format im Draft-Body bleibt unverändert
(Extractor/Verifier/anchor_repair/page_index lesen weiter `S. N`) — die
Kennzeichnung passiert ausschließlich beim Rendern.

Till-Entscheid 2026-07-07: Variante „Kennzeichnen", KEINE Offset-Heuristik.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from generative.pipeline import pdf_chunker
from generative.pipeline.citation_check import apply_physical_pages_flag
from generative.pipeline.note_json import note_to_json_dict
from generative.pipeline.portable_md import render_portable_note
from generative.pipeline.vault_writer import (
    build_quellen_block,
    convert_inline_to_footnotes,
    pages_from_body,
    render_note,
)
from generative.schemas.atomic_note import AtomicNoteDraft, QualityReport
from generative.schemas.citation import CitationMeta, build_citation_meta


def _qr(**kw) -> QualityReport:
    defaults = dict(peer_reviewed=None, citation_count=None, retracted=False, flags=[])
    defaults.update(kw)
    return QualityReport(**defaults)


def _draft(**overrides) -> AtomicNoteDraft:
    base = dict(
        title="Activation Phase",
        body="# Activation Phase\n\nErster Satz (S. 3).",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high",
    )
    base.update(overrides)
    return AtomicNoteDraft(**base)


def _citation(**overrides) -> CitationMeta:
    base = dict(author="Merrill", year="2002", title="First Principles", doi=None, source_file="merrill-2002.pdf")
    base.update(overrides)
    return CitationMeta(**base)


class TestPdfUsesPhysicalPages(unittest.TestCase):
    """`pdf_chunker.pdf_uses_physical_pages`: Signal, ob `pdf_to_pages` auf den
    `i+1`-Fallback zurückfällt (keine nutzbaren `/PageLabels`)."""

    def test_no_labels_is_physical(self):
        with patch("generative.pipeline.pdf_chunker._pdf_page_labels", return_value=None):
            self.assertTrue(pdf_chunker.pdf_uses_physical_pages(Path("x.pdf")))

    def test_labels_present_is_not_physical(self):
        with patch("generative.pipeline.pdf_chunker._pdf_page_labels", return_value=["51", "52"]):
            self.assertFalse(pdf_chunker.pdf_uses_physical_pages(Path("x.pdf")))


class TestCitationMetaPhysicalPagesField(unittest.TestCase):
    def test_default_is_false(self):
        c = CitationMeta(author="A", year="2020", title="T", doi=None, source_file="x.pdf")
        self.assertFalse(c.physical_pages)

    def test_explicit_true(self):
        c = CitationMeta(author="A", year="2020", title="T", doi=None, source_file="x.pdf", physical_pages=True)
        self.assertTrue(c.physical_pages)


class TestBuildCitationMetaPhysicalPages(unittest.TestCase):
    def test_default_false(self):
        c = build_citation_meta({"Author": "A", "Year": "2020", "Title": "T"}, _qr(), "T", "x.pdf")
        self.assertFalse(c.physical_pages)

    def test_true_forwarded(self):
        c = build_citation_meta({"Author": "A", "Year": "2020", "Title": "T"}, _qr(), "T", "x.pdf", physical_pages=True)
        self.assertTrue(c.physical_pages)


class TestFootnoteDefKennzeichnung(unittest.TestCase):
    """Footnote-Defs tragen `PDF-S.` statt `S.` wenn `physical_pages=True` —
    Pipeline-internes `(S. N)`-Inline-Format im Draft-Body bleibt unverändert
    (nur der Renderer sieht `physical_pages`)."""

    def test_plaintext_def_marked(self):
        out = convert_inline_to_footnotes("Satz (S. 3).", "Merrill 2002", physical_pages=True)
        self.assertIn("[^1]: Merrill 2002, PDF-S. 3.", out)

    def test_plaintext_def_unmarked_by_default(self):
        out = convert_inline_to_footnotes("Satz (S. 3).", "Merrill 2002")
        self.assertIn("[^1]: Merrill 2002, S. 3.", out)
        self.assertNotIn("PDF-S.", out)

    @patch("generative.pipeline.vault_writer.LITERATURE_DIR")
    def test_wikilink_def_marks_label_keeps_page_fragment_bare(self, mock_lit):
        # Das `#page=N`-Fragment MUSS die nackte Zahl bleiben (Obsidian-PDF-Viewer
        # braucht den physischen Index — der stimmt hier gerade), nur das Label
        # (Anzeigetext) wird gekennzeichnet.
        mock_path = type("P", (), {"exists": lambda self: True})()
        mock_lit.__truediv__ = lambda self, other: mock_path
        out = convert_inline_to_footnotes(
            "Satz (S. 3).", "Merrill 2002", source_file="merrill.pdf", physical_pages=True
        )
        self.assertIn("[[merrill.pdf#page=3|PDF-S. 3]]", out)


class TestQuellenBlockKennzeichnung(unittest.TestCase):
    def test_marked_when_physical(self):
        body = "Text[^1].\n\n[^1]: Merrill 2002, PDF-S. 3."
        citation = _citation(physical_pages=True)
        out = build_quellen_block(body, "merrill-2002.pdf", citation)
        self.assertIn(", PDF-S. 3*", out)

    def test_unmarked_when_labels_present(self):
        body = "Text[^1].\n\n[^1]: Merrill 2002, S. 3."
        citation = _citation(physical_pages=False)
        out = build_quellen_block(body, "merrill-2002.pdf", citation)
        self.assertIn(", S. 3*", out)
        self.assertNotIn("PDF-S.", out)


class TestPagesFromBodyUnderstandsPdfLabel(unittest.TestCase):
    """Kohärenz-Wächter aus #138 erweitert: `pages_from_body` muss die
    gekennzeichnete `PDF-S. N`-Form genauso verstehen wie `S. N` — sonst
    verliert der Quellen-Block bei gekennzeichneten Notes seine Seiten."""

    def test_plaintext_def_pdf_label(self):
        body = "Text[^1].\n\n[^1]: Merrill 2002, PDF-S. 3."
        self.assertEqual(pages_from_body(body), ["3"])

    def test_wikilink_def_pdf_label(self):
        body = "Text[^1].\n\n[^1]: Merrill 2002, [[merrill.pdf#page=3|PDF-S. 3]]."
        self.assertEqual(pages_from_body(body), ["3"])

    def test_wikilink_def_pdf_label_range(self):
        body = "Text[^1].\n\n[^1]: Merrill 2002, [[merrill.pdf#page=13|PDF-S. 13–14]]."
        self.assertEqual(pages_from_body(body), ["13–14"])

    def test_mixed_marked_and_unmarked_defs_dedup(self):
        body = "A[^1] B[^2].\n\n[^1]: X, PDF-S. 3.\n[^2]: X, PDF-S. 3."
        self.assertEqual(pages_from_body(body), ["3"])


class TestRenderNoteEndToEnd(unittest.TestCase):
    """Golden-Vergleich (Leitplanke 4): `physical_pages=False` (Labels vorhanden)
    -> exakt heutiges Verhalten, keine Kennzeichnung."""

    def test_physical_pages_true_marks_footnote_and_quellen_block(self):
        draft = _draft(body="# Activation Phase\n\nErster Satz (S. 3).")
        out = render_note(draft, "merrill-2002.pdf", citation=_citation(physical_pages=True))
        self.assertIn("[^1]: Merrill 2002, PDF-S. 3.", out)
        self.assertIn("*Quelle: Merrill 2002: First Principles, PDF-S. 3*", out)

    def test_physical_pages_false_matches_current_behavior(self):
        draft = _draft(body="# Activation Phase\n\nErster Satz (S. 3).")
        out = render_note(draft, "merrill-2002.pdf", citation=_citation(physical_pages=False))
        self.assertIn("[^1]: Merrill 2002, S. 3.", out)
        self.assertIn("*Quelle: Merrill 2002: First Principles, S. 3*", out)
        self.assertNotIn("PDF-S.", out)


class TestNoteJsonPhysicalPagesExport(unittest.TestCase):
    """F1-Export-Contract (`note_json._source_block`) trägt `physical_pages`
    additiv mit — kein SCHEMA_VERSION-Bump (rein additives Feld)."""

    def test_physical_pages_true_in_source_citation(self):
        result = note_to_json_dict(_draft(), _citation(physical_pages=True))
        self.assertTrue(result["source"]["citation"]["physical_pages"])

    def test_physical_pages_false_in_source_citation(self):
        result = note_to_json_dict(_draft(), _citation(physical_pages=False))
        self.assertFalse(result["source"]["citation"]["physical_pages"])


class TestPortableMdKennzeichnung(unittest.TestCase):
    def test_physical_pages_marks_footnote_and_quellen(self):
        draft = _draft(body="# Activation Phase\n\nErster Satz (S. 3).")
        note_json = note_to_json_dict(draft, _citation(physical_pages=True))
        out = render_portable_note(note_json)
        self.assertIn("[^1]: Merrill 2002, PDF-S. 3.", out)
        self.assertIn("Merrill 2002: First Principles, PDF-S. 3", out)

    def test_physical_pages_false_matches_current_behavior(self):
        draft = _draft(body="# Activation Phase\n\nErster Satz (S. 3).")
        note_json = note_to_json_dict(draft, _citation(physical_pages=False))
        out = render_portable_note(note_json)
        self.assertIn("[^1]: Merrill 2002, S. 3.", out)
        self.assertNotIn("PDF-S.", out)


class TestApplyPhysicalPagesFlag(unittest.TestCase):
    """Seiteneffekt-freier Review-Hinweis (analog zu `apply_citation_check`):
    Quality-Flag an allen Drafts einer Quelle ohne `/PageLabels`."""

    def test_flags_all_drafts_when_physical(self):
        drafts = [_draft(title="A"), _draft(title="B")]
        citation = _citation(physical_pages=True)
        added = apply_physical_pages_flag(drafts, citation)
        self.assertEqual(added, 2)
        for d in drafts:
            self.assertTrue(any("PDF-Position" in f for f in d.quality_flags))

    def test_no_flag_when_labels_present(self):
        drafts = [_draft()]
        citation = _citation(physical_pages=False)
        added = apply_physical_pages_flag(drafts, citation)
        self.assertEqual(added, 0)
        self.assertEqual(drafts[0].quality_flags, [])

    def test_idempotent(self):
        drafts = [_draft()]
        citation = _citation(physical_pages=True)
        apply_physical_pages_flag(drafts, citation)
        apply_physical_pages_flag(drafts, citation)
        self.assertEqual(len(drafts[0].quality_flags), 1)


class TestOrchestratorBuildCitationWiring(unittest.TestCase):
    """`_build_citation` reicht `physical_pages` an `build_citation_meta` durch."""

    def test_build_citation_forwards_physical_pages(self):
        from generative import orchestrator

        citation = orchestrator._build_citation(
            {"Author": "A", "Year": "2020", "Title": "T"}, _qr(), "T", "x.pdf", physical_pages=True
        )
        self.assertTrue(citation.physical_pages)

    def test_build_citation_defaults_to_false(self):
        from generative import orchestrator

        citation = orchestrator._build_citation({"Author": "A", "Year": "2020", "Title": "T"}, _qr(), "T", "x.pdf")
        self.assertFalse(citation.physical_pages)


if __name__ == "__main__":
    unittest.main()
