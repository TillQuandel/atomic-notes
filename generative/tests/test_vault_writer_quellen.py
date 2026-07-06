"""Tests für Issue #76: Quellen-Block-Seiten aus dem final gerenderten Body statt
aus `note.source_anchors`.

Wurzel: `source_anchors` sind der Verifier-Stand VOR Critic/Layout/Renumber —
diese Stufen können Fußnoten aus dem Body entfernen (Layout-Refactor,
Hub-Redundanz-Filter) oder ergänzen (Verifier-Nachtrag), ohne `source_anchors`
nachzuziehen. Zwei belegte Drift-Klassen:
- Phantom-Seite: ein Anker (inkl. `fuzzy_page`) ohne zugehörige Body-Fußnote
  taucht trotzdem im Quellen-Block auf.
- Fehlende Seite: eine im Body verbliebene/ergänzte Fußnote hat keinen
  passenden Anker mehr und fehlt im Quellen-Block.

`pages_from_body` ersetzt `source_anchors` als Quelle für den Quellen-Block:
Seiten werden direkt aus dem übergebenen Body-Text gelesen (Footnote-Defs
`[^n]: ...` in beiden Formaten, oder — für den Vor-Konvertierungs-Zustand —
Inline-`(S. N)`-Anker).
"""

import re
import unittest

from generative.pipeline.note_json import note_to_json_dict
from generative.pipeline.portable_md import render_portable_note
from generative.pipeline.vault_writer import (
    build_quellen_block,
    pages_from_body,
    render_moc,
    render_note,
)
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor
from generative.schemas.citation import CitationMeta


def _draft(**overrides) -> AtomicNoteDraft:
    base = dict(
        title="Test-Konzept",
        body="Body ohne Anker.",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high",
    )
    base.update(overrides)
    return AtomicNoteDraft(**base)


def _citation(**overrides) -> CitationMeta:
    base = dict(author="Autor", year="2020", title="Titel", doi=None, source_file="autor-2020.pdf")
    base.update(overrides)
    return CitationMeta(**base)


class TestPagesFromBodyFootnoteDefs(unittest.TestCase):
    """`pages_from_body` auf bereits footnote-konvertierten Bodies (Klartext-
    und Wikilink-Def-Form)."""

    def test_plaintext_def_single_page(self):
        body = "Text[^1].\n\n[^1]: Autor 2020, S. 13."
        self.assertEqual(pages_from_body(body), ["13"])

    def test_plaintext_def_range(self):
        body = "Text[^1].\n\n[^1]: Autor 2020, S. 159–160."
        self.assertEqual(pages_from_body(body), ["159–160"])

    def test_plaintext_def_comma_list_splits_into_tokens(self):
        body = "Text[^1].\n\n[^1]: Autor 2020, S. 13, 15."
        self.assertEqual(pages_from_body(body), ["13", "15"])

    def test_wikilink_def_single_page(self):
        body = "Text[^1].\n\n[^1]: Autor 2020, [[Datei.pdf#page=13|S. 13]]."
        self.assertEqual(pages_from_body(body), ["13"])

    def test_wikilink_def_range(self):
        body = "Text[^1].\n\n[^1]: Autor 2020, [[Datei.pdf#page=13|S. 13–14]]."
        self.assertEqual(pages_from_body(body), ["13–14"])

    def test_multiple_defs_dedup_and_numeric_sort(self):
        body = "A[^1] B[^2] C[^3].\n\n[^1]: X, S. 159.\n[^2]: X, S. 9.\n[^3]: X, S. 159."
        self.assertEqual(pages_from_body(body), ["9", "159"])


class TestPagesFromBodyInlineAnchors(unittest.TestCase):
    """`pages_from_body` auf Vor-Konvertierungs-Bodies (rohe `(S. N)`-Anker)."""

    def test_single_inline_anchor(self):
        self.assertEqual(pages_from_body("Ein Satz (S. 8)."), ["8"])

    def test_multiple_inline_anchors_dedup_sort(self):
        body = "Erst (S. 9). Dann (S. 2). Nochmal (S. 9)."
        self.assertEqual(pages_from_body(body), ["2", "9"])

    def test_inline_range(self):
        self.assertEqual(pages_from_body("Satz (S. 13-14)."), ["13–14"])

    def test_inline_comma_list(self):
        self.assertEqual(pages_from_body("Satz (S. 13, 15)."), ["13", "15"])

    def test_blockquote_lines_ignored(self):
        body = "> [!quote]- Autor 2020, S. 99\n> Zitat aus der Quelle."
        self.assertEqual(pages_from_body(body), [])

    def test_empty_body_yields_empty_list(self):
        self.assertEqual(pages_from_body("Kein Anker hier."), [])

    def test_mixed_def_and_inline_dedup(self):
        body = "Vorn (S. 9).\n\n[^1]: X, S. 159.\n[^2]: X, S. 9."
        self.assertEqual(pages_from_body(body), ["9", "159"])


class TestIssuePhantomPage(unittest.TestCase):
    """Issue-Fall 1 (Phantom-Seite): fuzzy_page-Anker ohne Body-Fußnote darf
    NICHT mehr im Quellen-Block auftauchen."""

    def test_fuzzy_anchor_without_body_footnote_not_in_quellen_block(self):
        draft = _draft(
            body="# T\n\nErster Satz (S. 1). Zweiter Satz (S. 2).",
            source_anchors=[
                TextAnchor(quote="a", page="S. 1"),
                TextAnchor(quote="b", page="S. 2"),
                TextAnchor(quote="c", page=None, fuzzy_page="S. 8"),  # Phantom: keine Body-Fußnote
            ],
        )
        out = render_note(draft, "autor-2020.pdf", citation=_citation())
        quellen = out.split("## Quellen", 1)[1]
        self.assertIn("S. 1, 2", quellen)
        self.assertNotIn("8", quellen)


class TestIssueMissingPage(unittest.TestCase):
    """Issue-Fall 2 (fehlende Seite): Body-Fußnote S. 166 zusätzlich zu 159/160,
    source_anchors kennt die 166 nicht (z.B. nachträglich vom Verifier ergänzt
    oder von einer Nachbearbeitungsstufe eingefügt) — Quellen-Block muss die
    166 trotzdem zeigen."""

    def test_extra_body_footnote_page_appears_in_quellen_block(self):
        draft = _draft(
            body=(
                "# T\n\nErster Satz (S. 159). Zweiter Satz (S. 160). Dritter, nachtraeglich ergaenzter Satz (S. 166)."
            ),
            source_anchors=[
                TextAnchor(quote="a", page="S. 159"),
                TextAnchor(quote="b", page="S. 160"),
                # 166 fehlt bewusst in source_anchors
            ],
        )
        out = render_note(draft, "autor-2020.pdf", citation=_citation())
        quellen = out.split("## Quellen", 1)[1]
        self.assertIn("S. 159, 160, 166", quellen)


class TestNoFallbackToSourceAnchors(unittest.TestCase):
    """Body ganz ohne Seiten-Beleg -> leerer Seiten-Marker, KEIN Fallback auf
    source_anchors (der würde die Phantom-Klasse aus Issue #76 re-öffnen)."""

    def test_no_anchors_in_body_no_fallback(self):
        draft = _draft(
            body="# T\n\nSatz ganz ohne Seiten-Anker.",
            source_anchors=[TextAnchor(quote="a", page="S. 5")],
        )
        out = render_note(draft, "autor-2020.pdf", citation=_citation())
        quellen = out.split("## Quellen", 1)[1]
        self.assertNotIn(", S.", quellen)


class TestRenderNoteCoherence(unittest.TestCase):
    """Kohärenz-Property (die eigentliche Issue-Forderung): die im Quellen-Block
    gelisteten Seiten müssen exakt den Seiten der gerenderten Footnote-Defs
    entsprechen — unabhängig davon, was source_anchors sagt."""

    def test_quellen_pages_match_rendered_footnote_def_pages(self):
        draft = _draft(
            body="# T\n\nA (S. 3). B (S. 3). C (S. 21).",
            source_anchors=[TextAnchor(quote="x", page="S. 999")],  # bewusst veraltet/falsch
        )
        out = render_note(draft, "autor-2020.pdf", citation=_citation())
        def_pages = set(re.findall(r"\[\^\d+\]: [^,]+, S\. ([\d–]+)\.", out))
        quellen_line = next(line for line in out.splitlines() if line.startswith("*Quelle:"))
        m = re.search(r", S\. (.+)\*$", quellen_line)
        quellen_pages = {p.strip() for p in m.group(1).split(",")} if m else set()
        self.assertEqual(def_pages, quellen_pages)
        self.assertNotIn("999", out)


class TestRenderNoteCoherencePhysicalPages(unittest.TestCase):
    """Issue #95: dieselbe Kohärenz-Property wie `TestRenderNoteCoherence`, aber
    mit `physical_pages=True` (PDF ohne `/PageLabels`) — die gekennzeichnete
    `PDF-S.`-Form darf die Seiten-Kohärenz zwischen Footnote-Defs und Quellen-
    Block nicht brechen (`pages_from_body` muss beide Formen verstehen)."""

    def test_quellen_pages_match_rendered_footnote_def_pages_when_marked(self):
        draft = _draft(
            body="# T\n\nA (S. 3). B (S. 3). C (S. 21).",
            source_anchors=[TextAnchor(quote="x", page="S. 999")],  # bewusst veraltet/falsch
        )
        out = render_note(draft, "autor-2020.pdf", citation=_citation(physical_pages=True))
        def_pages = set(re.findall(r"\[\^\d+\]: [^,]+, PDF-S\. ([\d–]+)\.", out))
        quellen_line = next(line for line in out.splitlines() if line.startswith("*Quelle:"))
        m = re.search(r", PDF-S\. (.+)\*$", quellen_line)
        quellen_pages = {p.strip() for p in m.group(1).split(",")} if m else set()
        self.assertEqual(def_pages, quellen_pages)
        self.assertEqual(def_pages, {"3", "21"})
        self.assertNotIn("999", out)


class TestRenderMocUsesFinalBody(unittest.TestCase):
    """render_moc kann Absätze (und damit Fußnoten) nach der Konvertierung noch
    über den Hub-Redundanz-Filter + renumber_footnotes verlieren — der Quellen-
    Block muss den tatsächlichen body_combined widerspiegeln, nicht
    source_anchors."""

    def test_moc_quellen_pages_match_final_body_not_source_anchors(self):
        draft = _draft(
            title="Hub-Titel",
            body="# Hub\n\nEinleitung (S. 4).",
            source_anchors=[TextAnchor(quote="a", page="S. 4"), TextAnchor(quote="b", page="S. 77")],
            action="hub",
            hub_subconcepts=["A", "B"],
        )
        out = render_moc(draft, "autor-2020.pdf", citation=_citation())
        quellen = out.split("## Quellen", 1)[1]
        self.assertIn("S. 4", quellen)
        self.assertNotIn("77", quellen)


class TestPortableMdCoherence(unittest.TestCase):
    """portable_md._render_quellen_section (via render_portable_note) muss
    dieselbe Body-basierte Logik nutzen wie vault_writer."""

    def test_portable_quellen_pages_match_body_not_source_anchors(self):
        draft = _draft(
            title="T",
            body="# T\n\nA (S. 4). B (S. 9).",
            source_anchors=[TextAnchor(quote="a", page="S. 999")],  # bewusst veraltet/falsch
        )
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json)
        self.assertIn("S. 4, 9", out)
        self.assertNotIn("999", out)


class TestBuildQuellenBlockTakesBody(unittest.TestCase):
    """build_quellen_block liest Seiten nur noch aus dem übergebenen Body-Text,
    nicht mehr aus einem note-Objekt."""

    def test_pages_come_from_body_param(self):
        body = "Text[^1].\n\n[^1]: Autor 2020, S. 42."
        out = build_quellen_block(body, "autor-2020.pdf", _citation())
        self.assertIn(", S. 42*", out)

    def test_page_prefix_not_doubled(self):
        body = "Text[^1].\n\n[^1]: Autor 2020, S. 1."
        out = build_quellen_block(body, "autor-2020.pdf", _citation())
        self.assertIn(", S. 1*", out)
        self.assertNotIn("S. S.", out)
