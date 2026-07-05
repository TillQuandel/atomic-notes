"""Tests für pipeline.portable_md — portabler Markdown-Renderer aus note.json
(Output-Projekt F2). Konsumiert den F1-Export-Contract (`note_to_json_dict`/
`run_to_json_dict`) und erzeugt CommonMark+Standard-Footnotes ohne Obsidian-
Spezifika (kein YAML-Frontmatter, kein `[[Wikilink]]`, keine Callouts).
"""

import copy
import unittest

from generative.pipeline.note_json import note_to_json_dict, run_to_json_dict
from generative.pipeline.portable_md import (
    gfm_anchor_slug,
    render_portable_note,
    render_portable_run,
)
from generative.pipeline.vault_writer import collect_anchor_pages
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor
from generative.schemas.citation import CitationMeta


def _draft(**overrides) -> AtomicNoteDraft:
    base = dict(
        title="Atomic Notes",
        body="# Atomic Notes\n\nEine Note enthält eine Idee (S. 1).",
        source_anchors=[TextAnchor(quote="q", page="S. 1")],
        related=[],
        tags=["zettelkasten"],
        synthesis_confidence="low",
    )
    base.update(overrides)
    return AtomicNoteDraft(**base)


def _citation(**overrides) -> CitationMeta:
    base = dict(author=None, year=None, title=None, doi=None, source_file="zettelkasten-primer.pdf")
    base.update(overrides)
    return CitationMeta(**base)


class TestHeadRendering(unittest.TestCase):
    """Regel 2: kein YAML-Frontmatter, H1 + kursive Quellzeile statt Kopfzeilen-Block."""

    def test_no_yaml_h1_and_source_line(self):
        draft = _draft()
        citation = _citation(
            author="Bates",
            year="2017",
            title="Information Behavior",
            source_file="Bates - 2017 - Information Behavior.pdf",
        )
        note_json = note_to_json_dict(draft, citation)
        out = render_portable_note(note_json)
        self.assertFalse(out.startswith("---"))
        self.assertTrue(out.startswith("# Atomic Notes"))
        self.assertIn(
            "*Quelle: Bates, 2017 — Information Behavior (Bates - 2017 - Information Behavior.pdf)*",
            out,
        )

    def test_source_line_missing_author_uses_short_label_no_none(self):
        draft = _draft()
        citation = _citation()  # author=None, title=None -> short_label faellt auf Dateiname-Stem zurueck
        note_json = note_to_json_dict(draft, citation)
        out = render_portable_note(note_json)
        self.assertNotIn("None", out)
        self.assertIn("zettelkasten-primer", out)


class TestFootnoteConversion(unittest.TestCase):
    """Regel 1: Inline-`(S. N)` -> `[^i]`-Footnote via bestehendem
    convert_inline_to_footnotes (source_file=None erzwingt Klartext)."""

    def test_inline_anchor_becomes_plain_footnote(self):
        draft = _draft(body="# T\n\nSatz (S. 4).", source_anchors=[TextAnchor(quote="q", page="S. 4")])
        citation = _citation(author="Hiatt", year="2006", source_file="hiatt.pdf")
        note_json = note_to_json_dict(draft, citation)
        out = render_portable_note(note_json)
        self.assertIn("[^1]", out)
        self.assertIn("[^1]: Hiatt 2006, S. 4.", out)
        self.assertNotIn("[[", out)

    def test_existing_footnotes_skip_reconversion(self):
        body = "# T\n\nErster Satz[^1]. Zweiter Satz (S. 2).\n\n[^1]: Hiatt 2006, S. 1."
        draft = _draft(body=body)
        citation = _citation(author="Hiatt", year="2006")
        note_json = note_to_json_dict(draft, citation)
        out = render_portable_note(note_json)
        # Body traegt schon [^N] -> keine erneute Konvertierung, "(S. 2)" bleibt unangetastet
        self.assertIn("(S. 2)", out)
        self.assertNotIn("[^2]", out)


class TestCalloutConversion(unittest.TestCase):
    """Regel 3: Obsidian-Callout-Header -> Standard-Blockquote mit fettem Header."""

    def test_callout_with_header_becomes_bold_blockquote(self):
        body = '# T\n\nSatz (S. 1).\n\n> [!quote]- o. V. [o. J.], S. 1\n> „Zitat."'
        draft = _draft(body=body)
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json)
        self.assertIn("> **o. V. [o. J.], S. 1**", out)
        self.assertIn('> „Zitat."', out)
        self.assertNotIn("[!quote]", out)

    def test_callout_without_header_strips_header_line(self):
        body = "# T\n\nSatz (S. 1).\n\n> [!note]\n> Weiterer Text."
        draft = _draft(body=body)
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json)
        self.assertNotIn("[!note]", out)
        self.assertIn("> Weiterer Text.", out)


class TestWikilinkResolution(unittest.TestCase):
    """Regel 4: `[[Ziel]]`/`[[Ziel|Anzeige]]`/Fragment-Formen; auflösbar -> Link,
    sonst Klartext-Anzeige."""

    def test_internal_link_file_mode(self):
        draft = _draft(body="# T\n\nSiehe [[Atomic Notes]] (S. 1).")
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json, exported_titles={"atomic notes": "Atomic Notes"}, link_mode="file")
        self.assertIn("[Atomic Notes](Atomic%20Notes.md)", out)

    def test_internal_link_anchor_mode(self):
        draft = _draft(body="# T\n\nSiehe [[Atomic Notes]] (S. 1).")
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(
            note_json,
            exported_titles={"atomic notes": "Atomic Notes: Eine Note enthält genau eine Idee"},
            link_mode="anchor",
        )
        self.assertIn("[Atomic Notes](#atomic-notes-eine-note-enthält-genau-eine-idee)", out)

    def test_unresolvable_link_becomes_plain_text(self):
        draft = _draft(body="# T\n\nSiehe [[Fremde Note]] (S. 1).")
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json, exported_titles={})
        self.assertIn("Siehe Fremde Note", out)
        self.assertNotIn("[[", out)
        self.assertNotIn("]]", out)

    def test_external_alias_link_becomes_alias_text(self):
        draft = _draft(body="# T\n\nSiehe [[Ziel|Anzeige]] (S. 1).")
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json, exported_titles={})
        self.assertIn("Siehe Anzeige", out)

    def test_pdf_page_fragment_link_becomes_alias_text(self):
        draft = _draft(body="# T\n\nSiehe [[pdf.pdf#page=3|S. 3]].")
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json, exported_titles={})
        self.assertIn("Siehe S. 3", out)
        self.assertNotIn("[[", out)


class TestGfmAnchorSlug(unittest.TestCase):
    """Regel 5: pandoc/GFM-kompatible Anchor-Slugs."""

    def test_heading_with_subtitle(self):
        self.assertEqual(
            gfm_anchor_slug("Progressive Summarization: Schichtweise Verdichtung"),
            "progressive-summarization-schichtweise-verdichtung",
        )

    def test_umlaut_preserved(self):
        self.assertEqual(gfm_anchor_slug("Übersicht"), "übersicht")


class TestRenderPortableRun(unittest.TestCase):
    """Regel 6: Sammel-Dokument mit Footnote-Offset pro Note + `---`-Trenner +
    funktionierender interner Anker-Link."""

    def test_two_notes_offset_separator_and_internal_anchor(self):
        draft1 = _draft(
            title="Atomic Notes",
            body="# Atomic Notes\n\nErste Idee (S. 1). Zweite Idee (S. 2).",
            source_anchors=[TextAnchor(quote="a", page="S. 1"), TextAnchor(quote="b", page="S. 2")],
        )
        draft2 = _draft(
            title="Progressive Summarization",
            body="# Progressive Summarization\n\nSiehe [[Atomic Notes]] (S. 1).",
            source_anchors=[TextAnchor(quote="c", page="S. 1")],
        )
        run_json = run_to_json_dict([draft1, draft2], _citation())
        out = render_portable_run(run_json)

        self.assertIn("\n\n---\n\n", out)
        self.assertIn("[^1]", out)
        self.assertIn("[^2]", out)
        # Note 2 beginnt bei [^3] (Offset = 2 Footnotes aus Note 1)
        self.assertIn("[^3]: ", out)
        self.assertNotIn("[^1]: ", out.split("---")[1])  # Note 2 hat keine eigene [^1]-Def mehr
        # interner Link auf Note 1 als funktionierender Anker (Slug von "# Atomic Notes")
        self.assertIn("(#atomic-notes)", out)

    def test_offset_footnotes_orphan_def_no_crash(self):
        # Verifizierter Review-Fund (PR #134): eine Def ohne Marker (Orphan,
        # möglich auf dem „Body bereits konvertiert"-Pfad) crashte mit KeyError.
        # Orphan-Defs werden mitverschoben — eindeutig nummeriert, keine
        # Kollision mit der Folge-Note.
        from generative.pipeline.portable_md import _offset_footnotes

        text = "Text[^1].\n\n[^1]: def eins.\n[^9]: orphan def."
        out, offset = _offset_footnotes(text, 2)
        self.assertIn("[^3]", out)  # Marker 1 + Offset 2
        self.assertIn("[^11]: orphan def.", out)  # Orphan 9 + Offset 2
        self.assertEqual(offset, 11)  # Folge-Note startet nach der Orphan-Def


class TestMetadataSection(unittest.TestCase):
    """Regel 7: Betriebsdaten standardmäßig verborgen, nur mit include_metadata=True."""

    def test_default_excludes_metadata(self):
        # bewusst tag-namens-disjunkt vom source_file "zettelkasten-primer.pdf",
        # sonst kollidiert die Quellzeile mit dem Substring-Check.
        draft = _draft(tags=["knowledge-management"], quality_flags=["⚠️ sonderflag-xyz"])
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json)
        self.assertNotIn("## Metadaten", out)
        self.assertNotIn("knowledge-management", out)
        self.assertNotIn("sonderflag-xyz", out)

    def test_include_metadata_true_renders_section_with_routing(self):
        draft = _draft(
            tags=["knowledge-management"], quality_flags=["⚠️ sonderflag-xyz"], critic_score=4, hard_gates_pass=True
        )
        note_json = note_to_json_dict(draft, _citation())
        out = render_portable_note(note_json, include_metadata=True)
        self.assertIn("## Metadaten", out)
        self.assertIn("knowledge-management", out)
        self.assertIn("sonderflag-xyz", out)
        self.assertIn("Vault-Empfehlung", out)


class TestQuellenSection(unittest.TestCase):
    """Regel 8: deterministischer Quellen-Absatz aus source_anchors (page/fuzzy_page,
    dedupliziert, numerisch sortiert)."""

    def test_pages_deduplicated_and_sorted(self):
        draft = _draft(
            source_anchors=[
                TextAnchor(quote="a", page="S. 159"),
                TextAnchor(quote="b", page="S. 9"),
                TextAnchor(quote="c", page="S. 159–160"),
                TextAnchor(quote="d", page=None, fuzzy_page="S. 9"),  # Dedup gegen b
            ]
        )
        citation = _citation(author="Foo", year="2020", title="Foo Titel", source_file="foo.pdf")
        note_json = note_to_json_dict(draft, citation)
        out = render_portable_note(note_json)
        self.assertIn("## Quellen", out)
        self.assertIn("Foo 2020: Foo Titel, S. 9, 159, 159–160", out)


class TestCollectAnchorPagesHelper(unittest.TestCase):
    """`collect_anchor_pages` (vault_writer) — geteilter Helper, hier direkt getestet."""

    def test_dedup_sort_and_fuzzy_fallback(self):
        anchors = [
            TextAnchor(quote="a", page="S. 159"),
            TextAnchor(quote="b", page=None, fuzzy_page="S. 9"),
            TextAnchor(quote="c", page="S. 159–160"),
        ]
        self.assertEqual(collect_anchor_pages(anchors), ["9", "159", "159–160"])


class TestNoMutation(unittest.TestCase):
    def test_render_portable_note_does_not_mutate_input(self):
        draft = _draft()
        note_json = note_to_json_dict(draft, _citation())
        before = copy.deepcopy(note_json)
        render_portable_note(note_json, exported_titles={"x": "y"}, include_metadata=True)
        self.assertEqual(note_json, before)

    def test_render_portable_run_does_not_mutate_input(self):
        drafts = [_draft(title="A"), _draft(title="B", body="# B\n\nText (S. 1).")]
        run_json = run_to_json_dict(drafts, _citation())
        before = copy.deepcopy(run_json)
        render_portable_run(run_json)
        self.assertEqual(run_json, before)


if __name__ == "__main__":
    unittest.main()
