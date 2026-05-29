"""Tests für pipeline.vault_writer.convert_inline_to_footnotes (v30/v31).

Coverage-Schwerpunkt:
- Page-Range mit Hyphen vs. En-Dash
- Wikilink-Generierung wenn PDF im Vault auflösbar
- Klartext-Fallback wenn PDF nicht auflösbar oder Filename-Sonderzeichen
"""
import unittest
from unittest.mock import patch
from pathlib import Path

from pipeline.vault_writer import convert_inline_to_footnotes, build_quellen_block
from schemas.atomic_note import AtomicNoteDraft, TextAnchor


def _draft_with_anchors(anchors: list[TextAnchor]) -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title="Test-Konzept",
        body="Body.",
        source_anchors=anchors,
        related=[],
        tags=[],
        synthesis_confidence="high",
    )


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


class TestQuellenBlockPagePrefix(unittest.TestCase):
    """Issue #20: Anker-Page-Werte enthalten bereits `S. ` (Verifier setzt
    `page_str = f"S. {n}"`). Der Quellen-Block darf den Prefix nicht erneut
    voranstellen — sonst `S. S. 1`."""

    SRC = "Bates - 2017 - Information Behavior.pdf"
    META = {"Author": "Bates", "Year": "2017", "Title": "Information Behavior"}

    def test_page_with_prefix_not_doubled(self):
        draft = _draft_with_anchors([TextAnchor(quote="x", page="S. 1")])
        out = build_quellen_block(draft, self.SRC, self.META)
        self.assertIn(", S. 1*", out)
        self.assertNotIn("S. S.", out)

    def test_fuzzy_page_with_prefix_not_doubled(self):
        draft = _draft_with_anchors([TextAnchor(quote="x", page=None, fuzzy_page="S. 2")])
        out = build_quellen_block(draft, self.SRC, self.META)
        self.assertIn(", S. 2*", out)
        self.assertNotIn("S. S.", out)

    def test_page_without_prefix_still_renders(self):
        # Extractor-Pfad kann rohe "page"-Werte ohne Prefix liefern.
        draft = _draft_with_anchors([TextAnchor(quote="x", page="3")])
        out = build_quellen_block(draft, self.SRC, self.META)
        self.assertIn(", S. 3*", out)
        self.assertNotIn("S. S.", out)

    def test_multiple_pages_each_stripped(self):
        draft = _draft_with_anchors([
            TextAnchor(quote="a", page="S. 1"),
            TextAnchor(quote="b", page="S. 5"),
        ])
        out = build_quellen_block(draft, self.SRC, self.META)
        self.assertIn(", S. 1, 5*", out)
        self.assertNotIn("S. S.", out)


class TestRewriteMergedRelatedLinks(unittest.TestCase):
    """Issue #21: Drafts, die beim Schreiben zu Merge-Stubs werden (Title-/Alias-
    Match im Vault), erscheinen unter dem Dateinamen der bestehenden Note. Sibling-
    Drafts behalten aber `related: [[<alter-draft-titel>]]` → toter Link. Pre-Pass
    schreibt diese related-Einträge auf das Merge-Target um."""

    def _draft(self, title, related=None, aliases=None):
        return AtomicNoteDraft(
            title=title, body="b", source_anchors=[],
            related=list(related or []), tags=[], synthesis_confidence="high",
            aliases=list(aliases or []),
        )

    def test_sibling_related_rewritten_to_merge_target(self):
        merged = self._draft("Information Behavior (Bates)")
        sibling = self._draft("Forschungsstroeme",
                              related=["[[Information Behavior (Bates)]]"])
        existing = {"information behavior (bates)":
                    "04-wissen/IBI Forschungsbereich Information Behavior.md"}
        from pipeline.vault_writer import rewrite_merged_related_links
        count = rewrite_merged_related_links([merged, sibling], existing)
        self.assertEqual(sibling.related,
                         ["[[IBI Forschungsbereich Information Behavior]]"])
        self.assertEqual(count, 1)

    def test_non_merged_related_untouched(self):
        from pipeline.vault_writer import rewrite_merged_related_links
        sibling = self._draft("A", related=["[[Some Other Note]]"])
        existing = {"information behavior (bates)":
                    "04-wissen/IBI Forschungsbereich Information Behavior.md"}
        count = rewrite_merged_related_links([sibling], existing)
        self.assertEqual(sibling.related, ["[[Some Other Note]]"])
        self.assertEqual(count, 0)

    def test_alias_match_also_rewritten(self):
        from pipeline.vault_writer import rewrite_merged_related_links
        merged = self._draft("Information Behavior (Bates)",
                             aliases=["IB Bates"])
        sibling = self._draft("B", related=["[[IB Bates]]"])
        existing = {"information behavior (bates)":
                    "04-wissen/IBI Forschungsbereich Information Behavior.md"}
        rewrite_merged_related_links([merged, sibling], existing)
        self.assertEqual(sibling.related,
                         ["[[IBI Forschungsbereich Information Behavior]]"])

    def test_display_alias_link_rewritten_to_canonical(self):
        from pipeline.vault_writer import rewrite_merged_related_links
        merged = self._draft("Information Behavior (Bates)")
        sibling = self._draft("C",
                             related=["[[Information Behavior (Bates)|IB]]"])
        existing = {"information behavior (bates)":
                    "04-wissen/IBI Forschungsbereich Information Behavior.md"}
        rewrite_merged_related_links([merged, sibling], existing)
        self.assertEqual(sibling.related,
                         ["[[IBI Forschungsbereich Information Behavior]]"])

    def test_no_existing_concepts_is_noop(self):
        from pipeline.vault_writer import rewrite_merged_related_links
        sibling = self._draft("D", related=["[[X]]"])
        self.assertEqual(rewrite_merged_related_links([sibling], None), 0)
        self.assertEqual(sibling.related, ["[[X]]"])


if __name__ == "__main__":
    unittest.main()
