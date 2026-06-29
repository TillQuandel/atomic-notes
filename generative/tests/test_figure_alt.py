"""Tests fuer figure_alt: deterministische Alt-Text-Einbettung aus getaggten PDFs.

Pfad C (Alt-Text-only): aus PDF-UA-getaggten PDFs den menschengeschriebenen
/Alt-Text der /Figure-Strukturelemente ziehen und an genau eine create-Note
binden (exakter source_anchor-Seitenmatch, nie fuzzy). Untagged-PDFs liefern
nichts (Gate). Empirisch motiviert in
docs/superpowers/specs/2026-06-04-figure-embedding-feasibility.md.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor

from generative.pipeline.figure_alt import (
    TaggedFigure,
    _sanitize_alt,
    bind_figures_to_drafts,
    embed_alt_figures,
    extract_tagged_figures,
    pdf_index_to_anchor_page,
)

FIXTURES = ROOT / "tests" / "fixtures"


def _draft(
    title: str, anchor_pages: list[str | None], action: str = "create", fuzzy_pages: list[str | None] | None = None
) -> AtomicNoteDraft:
    fuzzy_pages = fuzzy_pages or [None] * len(anchor_pages)
    anchors = [
        TextAnchor(quote=f"q{i}", page=p, fuzzy_page=fz) for i, (p, fz) in enumerate(zip(anchor_pages, fuzzy_pages))
    ]
    return AtomicNoteDraft(
        title=title,
        body=f"# {title}\n\nBody.",
        source_anchors=anchors,
        related=[],
        tags=[],
        synthesis_confidence="low",
        action=action,
    )


def test_sanitize_alt_collapses_whitespace_and_newlines():
    assert _sanitize_alt("Zeile eins\r\n  Zeile zwei") == "Zeile eins Zeile zwei"


def test_sanitize_alt_neutralizes_wikilinks_and_pipes():
    # /Alt ist untrusted: darf keine Obsidian-Wikilinks oder Tabellen erzeugen
    out = _sanitize_alt("siehe [[Konzept A]] und x | y")
    assert "[[" not in out and "]]" not in out
    assert "\\|" in out


def test_anchor_page_no_blank_pages_matches_pdf_index():
    # Alle Seiten haben Text → Anchor-Seite == 1-basierter PDF-Index
    flags = [True, True, True]
    assert pdf_index_to_anchor_page(flags, 2) == 3
    assert pdf_index_to_anchor_page(flags, 0) == 1


def test_anchor_page_blank_page_before_shifts_numbering():
    # Seite 1 (Index 1) textlos → pdftotext verwirft sie → Folge-Seiten verschieben
    flags = [True, False, True]
    assert pdf_index_to_anchor_page(flags, 2) == 2


def test_anchor_page_on_textless_page_is_unbindable():
    # Figur auf textloser Seite → kein source_anchor kann diese Seite je tragen
    flags = [True, False, True]
    assert pdf_index_to_anchor_page(flags, 1) is None


def test_anchor_page_uses_numeric_print_labels_when_present():
    # Mit /PageLabels (Buch) ist die Anker-Seite das Druckseiten-Label, nicht die
    # positionale Zählung — sonst matcht sie die source_anchors (jetzt Druckseiten)
    # nicht mehr (Regression durch den page-label-Fix).
    flags = [True, True, True]
    labels = ["159", "160", "161"]
    assert pdf_index_to_anchor_page(flags, 0, labels) == 159
    assert pdf_index_to_anchor_page(flags, 2, labels) == 161


def test_anchor_page_nonnumeric_label_falls_back_to_positional():
    # Römisches Frontmatter-Label (nicht-numerisch) → positionaler Fallback,
    # konsistent mit _resolve_page_numbers.
    flags = [True, True]
    labels = ["xi", "xii"]
    assert pdf_index_to_anchor_page(flags, 1, labels) == 2


def test_bind_unique_create_match_mutates_body():
    fig = TaggedFigure(anchor_page=3, alt_text="Ein Balkendiagramm zur Suche.", label="Abbildung 2")
    draft = _draft("Suche", ["S. 3"])

    report = bind_figures_to_drafts([fig], [draft])

    assert len(report.bound) == 1
    assert "Ein Balkendiagramm zur Suche." in draft.body
    assert "S. 3" in draft.body
    assert report.skipped == []


def test_bind_no_matching_page_is_skipped():
    fig = TaggedFigure(anchor_page=5, alt_text="x", label=None)
    draft = _draft("Suche", ["S. 3"])

    report = bind_figures_to_drafts([fig], [draft])

    assert report.bound == []
    assert len(report.skipped) == 1
    assert report.skipped[0].reason == "no_match"
    assert "Ein Balkendiagramm" not in draft.body


def test_bind_multiple_matches_is_ambiguous_skip():
    fig = TaggedFigure(anchor_page=3, alt_text="x", label=None)
    d1 = _draft("A", ["S. 3"])
    d2 = _draft("B", ["S. 3"])

    report = bind_figures_to_drafts([fig], [d1, d2])

    assert report.bound == []
    assert report.skipped[0].reason == "ambiguous"
    assert "x" not in d1.body and "x" not in d2.body


def test_bind_ignores_non_create_drafts():
    fig = TaggedFigure(anchor_page=3, alt_text="x", label=None)
    draft = _draft("Suche", ["S. 3"], action="extend")

    report = bind_figures_to_drafts([fig], [draft])

    assert report.bound == []
    assert report.skipped[0].reason == "no_match"


def test_bind_never_uses_fuzzy_page():
    # Anchor hat nur fuzzy_page="S. 3", page=None → darf NICHT matchen (precision-first)
    fig = TaggedFigure(anchor_page=3, alt_text="x", label=None)
    draft = _draft("Suche", [None], fuzzy_pages=["S. 3"])

    report = bind_figures_to_drafts([fig], [draft])

    assert report.bound == []
    assert report.skipped[0].reason == "no_match"


def test_bind_two_figures_same_note_grouped_under_one_heading():
    f1 = TaggedFigure(anchor_page=3, alt_text="erste Figur", label="Abbildung 1")
    f2 = TaggedFigure(anchor_page=3, alt_text="zweite Figur", label="Abbildung 2")
    draft = _draft("Suche", ["S. 3"])

    bind_figures_to_drafts([f1, f2], [draft])

    assert draft.body.count("## Abbildungen") == 1
    assert "erste Figur" in draft.body
    assert "zweite Figur" in draft.body


def test_extract_tagged_figures_reads_alt_and_page():
    figs = extract_tagged_figures(FIXTURES / "tagged_one_figure.pdf")
    assert len(figs) == 1
    page_index, alt = figs[0]
    assert page_index == 0
    assert alt == "Ein Saeulendiagramm der Suchhaeufigkeit."


def test_extract_tagged_figures_empty_on_untagged_pdf():
    assert extract_tagged_figures(FIXTURES / "untagged_plain.pdf") == []


def test_embed_alt_figures_binds_figure_on_tagged_pdf():
    # Fixture: 1 Textseite, Figur auf Seite 1 → anchor "S. 1"
    draft = _draft("Konzept A", ["S. 1"])

    report = embed_alt_figures(FIXTURES / "tagged_one_figure.pdf", [draft])

    assert len(report.bound) == 1
    assert "Saeulendiagramm" in draft.body
    assert "## Abbildungen" in draft.body


def test_embed_alt_figures_noop_on_untagged_pdf():
    draft = _draft("Konzept A", ["S. 1"])
    original = draft.body

    report = embed_alt_figures(FIXTURES / "untagged_plain.pdf", [draft])

    assert report.bound == []
    assert draft.body == original


def test_untagged_pdf_flagged_for_reporting():
    # #50/M11: untagged-PDF wird als untagged markiert, damit der Lauf den
    # Abbildungen-Skip einmal melden kann (statt stumm zu überspringen).
    report = embed_alt_figures(FIXTURES / "untagged_plain.pdf", [])
    assert report.untagged is True


def test_tagged_pdf_not_flagged_as_untagged():
    report = embed_alt_figures(FIXTURES / "tagged_one_figure.pdf", [_draft("Konzept A", ["S. 1"])])
    assert report.untagged is False
