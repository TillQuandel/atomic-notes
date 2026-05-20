"""Tests für pipeline.vault_writer.convert_inline_to_footnotes (v30/v31).

Coverage-Schwerpunkt:
- Page-Range mit Hyphen vs. En-Dash
- Wikilink-Generierung wenn PDF im Vault auflösbar
- Klartext-Fallback wenn PDF nicht auflösbar oder Filename-Sonderzeichen
"""
import unittest
from unittest.mock import patch
from pathlib import Path

from pipeline.vault_writer import convert_inline_to_footnotes


class TestPageRange(unittest.TestCase):
    """Page-Range-Erkennung im Inline-Marker `(S. N)`. Renderer muss Hyphen
    und En-Dash beide akzeptieren und auf En-Dash normalisieren."""

    def test_single_page(self):
        out = convert_inline_to_footnotes("Satz (S. 13).", "Hiatt 2006")
        self.assertIn("[^1]: Hiatt 2006, S. 13.", out)

    def test_page_range_hyphen(self):
        out = convert_inline_to_footnotes("Satz (S. 13-14).", "Hiatt 2006")
        self.assertIn("S. 13–14", out)

    def test_page_range_endash(self):
        out = convert_inline_to_footnotes("Satz (S. 13–14).", "Hiatt 2006")
        self.assertIn("S. 13–14", out)

    def test_page_range_with_spaces(self):
        out = convert_inline_to_footnotes("Satz (S. 13 - 14).", "Hiatt 2006")
        self.assertIn("S. 13–14", out)

    def test_page_comma_list(self):
        # Codex-Finding 1 (2026-05-10): (S. N, M)-Pattern muss erkannt werden
        out = convert_inline_to_footnotes("Satz (S. 13, 15).", "Hiatt 2006")
        self.assertIn("S. 13, 15", out)
        self.assertIn("[^1]", out)

    def test_page_comma_list_with_s_prefix(self):
        # (S. N, S. M)-Pattern (deutsche Direktzitat-Konvention)
        out = convert_inline_to_footnotes("Satz (S. 13, S. 15).", "Hiatt 2006")
        self.assertIn("S. 13, 15", out)


class TestWikilinkRendering(unittest.TestCase):
    """Wikilink-Variante (v30) wenn `source_file` und PDF im Vault auflösbar."""

    @patch("pipeline.vault_writer.LITERATURE_DIR")
    def test_wikilink_with_existing_pdf(self, mock_lit):
        # Mock: PDF existiert
        mock_path = type("P", (), {"exists": lambda self: True})()
        mock_lit.__truediv__ = lambda self, other: mock_path
        out = convert_inline_to_footnotes("Satz (S. 13).", "Hiatt 2006",
                                           source_file="Hiatt.pdf")
        self.assertIn("[[Hiatt.pdf#page=13|S. 13]]", out)

    @patch("pipeline.vault_writer.LITERATURE_DIR")
    def test_wikilink_page_range_uses_first_page(self, mock_lit):
        mock_path = type("P", (), {"exists": lambda self: True})()
        mock_lit.__truediv__ = lambda self, other: mock_path
        out = convert_inline_to_footnotes("Satz (S. 13–14).", "Hiatt 2006",
                                           source_file="Hiatt.pdf")
        # #page= zeigt auf erste Zahl, Label behält Range
        self.assertIn("[[Hiatt.pdf#page=13|S. 13–14]]", out)

    @patch("pipeline.vault_writer.LITERATURE_DIR")
    def test_klartext_fallback_when_pdf_missing(self, mock_lit):
        mock_path = type("P", (), {"exists": lambda self: False})()
        mock_lit.__truediv__ = lambda self, other: mock_path
        out = convert_inline_to_footnotes("Satz (S. 13).", "Hiatt 2006",
                                           source_file="missing.pdf")
        self.assertIn("S. 13.", out)
        self.assertNotIn("[[", out)

    def test_klartext_when_no_source_file(self):
        out = convert_inline_to_footnotes("Satz (S. 13).", "Hiatt 2006")
        self.assertIn("[^1]: Hiatt 2006, S. 13.", out)
        self.assertNotIn("[[", out)


class TestWikilinkUnsafeFilenames(unittest.TestCase):
    """v31 Codex-Finding 1: Filename mit Wikilink-Sonderzeichen darf keinen
    kaputten Wikilink produzieren — Klartext-Fallback."""

    @patch("pipeline.vault_writer.LITERATURE_DIR")
    def test_pipe_in_filename_falls_back(self, mock_lit):
        mock_path = type("P", (), {"exists": lambda self: True})()
        mock_lit.__truediv__ = lambda self, other: mock_path
        out = convert_inline_to_footnotes("Satz (S. 13).", "Author 2020",
                                           source_file="weird|name.pdf")
        self.assertNotIn("[[", out)
        self.assertIn("S. 13.", out)

    @patch("pipeline.vault_writer.LITERATURE_DIR")
    def test_hash_in_filename_falls_back(self, mock_lit):
        mock_path = type("P", (), {"exists": lambda self: True})()
        mock_lit.__truediv__ = lambda self, other: mock_path
        out = convert_inline_to_footnotes("Satz (S. 13).", "Author 2020",
                                           source_file="name#tag.pdf")
        self.assertNotIn("[[", out)

    @patch("pipeline.vault_writer.LITERATURE_DIR")
    def test_bracket_in_filename_falls_back(self, mock_lit):
        # Gemini-Finding G1: einfache Klammern brechen Wikilinks ebenfalls
        mock_path = type("P", (), {"exists": lambda self: True})()
        mock_lit.__truediv__ = lambda self, other: mock_path
        out = convert_inline_to_footnotes("Satz (S. 13).", "Author 2020",
                                           source_file="Studie [2023].pdf")
        self.assertNotIn("[[", out)
        self.assertIn("S. 13.", out)


class TestBlockQuotePreservation(unittest.TestCase):
    """Block-Quote-Callouts (`> ...`) dürfen NICHT zu Footnotes umgeschrieben
    werden — `S. N`-Angaben gehören dort zum Quote-Header."""

    def test_blockquote_not_converted(self):
        body = "> [!quote]- Hiatt 2006, S. 13\n> „..."
        out = convert_inline_to_footnotes(body, "Hiatt 2006")
        # Keine [^N] und kein [^N]: Definitions-Block
        self.assertNotIn("[^1]", out)


if __name__ == "__main__":
    unittest.main()
