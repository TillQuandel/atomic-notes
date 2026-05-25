"""Tests für pdf_chunker — Frontmatter-Strip + Page-Range-Helpers.

Frontmatter-Strip verhindert dass `concept_text_window` Cluster im PDF-Frontmatter
(Advance Praise, Acknowledgments, Copyright, …) findet, wenn Konzept-Begriffe
dort en passant vorkommen — Hiatt 2026-05-10 ADKAR-Eval-Bug. Siehe
[[Atomic-Agent-Pipeline]] v24.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.pdf_chunker import drop_frontmatter_pages, page_range_of_text, concept_text_window


# ---- drop_frontmatter_pages ---------------------------------------------

def test_drop_advance_praise_before_chapter_one():
    pages = [
        (1, "Advance Praise for ADKAR\n\n‹A model that revolutionizes change›\n— Reviewer A"),
        (2, "Copyright 2006. All rights reserved.\nISBN 978-1-9300-8505-0"),
        (3, "Acknowledgments\n\nThanks to the team for years of research."),
        (4, "1 Awareness\n\nAwareness of the need for change..."),
        (5, "More body content."),
        (6, "Even more body content."),
        (7, "End matter or final chapter section."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 3
    assert out[0][0] == 4  # Original-Page-Number bleibt
    assert "Awareness" in out[0][1]


def test_no_chapter_heading_no_crop():
    # Reines Tutorial ohne nummerierte Kapitel — nichts droppen.
    pages = [
        (1, "Step 1: Remove the bottom bracket."),
        (2, "Step 2: Clean threads with degreaser."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 0
    assert out == pages


def test_chapter_one_on_page_one_no_crop():
    pages = [
        (1, "1 Introduction\n\nThis book covers..."),
        (2, "More content."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 0
    assert out == pages


def test_no_frontmatter_signal_no_crop():
    # Pre-Pages ohne Frontmatter-Phrase → nicht aggressiv droppen.
    # Konservativ: Pre-Pages könnten Inhalt sein.
    pages = [
        (1, "Some paragraph without frontmatter markers about topic X."),
        (2, "Continued discussion of topic X over multiple pages."),
        (3, "1 First Real Chapter\n\nNow we begin properly."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 0


def test_cap_protects_against_overcrop():
    # Wenn erstes Chapter > 50% des Dokuments wegcroppen würde → kein Crop.
    pages = [
        (1, "Advance praise for the work."),
        (2, "Acknowledgments to colleagues."),
        (3, "Preface explaining motivation."),
        (4, "1 Chapter Begins Here\n\nMain content."),
    ]
    # 3/4 = 75% > 50% Cap → keine Veränderung
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 0


def test_german_frontmatter_recognized():
    # TOC-Eintrag wird ohne führende Ziffern formuliert, sonst triggert
    # _CHAPTER_RE bereits auf der TOC-Seite (Realismus-Limit der Heuristik —
    # echte TOCs mit Page-Number-Formatierung „1. Einführung 12" matchen
    # _CHAPTER_RE wegen des Punkts auch nicht).
    pages = [
        (1, "Vorwort\n\nDieses Buch entstand aus..."),
        (2, "Inhaltsverzeichnis"),
        (3, "1 Einführung\n\nWir beginnen mit den Grundlagen."),
        (4, "Mehr Inhalt."),
        (5, "Noch mehr Inhalt."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 2
    assert out[0][0] == 3


def test_short_doc_passthrough():
    pages = [(1, "Single page only.")]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 0
    assert out == pages


def test_empty_passthrough():
    out, dropped = drop_frontmatter_pages([])
    assert dropped == 0
    assert out == []


def test_isbn_alone_triggers_signal():
    # ISBN ist eindeutiges Frontmatter-Indiz auch ohne explizites „Copyright".
    pages = [
        (1, "Title Page"),  # kein Signal
        (2, "ISBN 978-3-16-148410-0\n\nPublisher info"),  # Signal hier
        (3, "1 Erste Kapitel-Section\n\nLos geht's."),
        (4, "Body."),
        (5, "More body."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 2
    assert out[0][0] == 3


# ---- page_range_of_text (Sanity) ----------------------------------------

def test_page_range_extracts_min_max():
    text = "[S. 5]\n\nfoo\n\n[S. 7]\n\nbar\n\n[S. 12]\n\nbaz"
    assert page_range_of_text(text) == (5, 12)


def test_page_range_no_marker_returns_none():
    assert page_range_of_text("just text") == (None, None)


# ---- concept_text_window: Co-Occurrence-Ranking (Option D, 2026-05-17) ------
# Vorgängerversion expandierte ±window_words um Treffer-Cluster; bei generischen
# Tokens (`agent`, `system`) wuchs der Cluster über das ganze Dokument und der
# Extractor sah nur die ersten 8000 chars (TOC+Intro). Neue Logik: 50%-Stride-
# Sliding-Window, Score = 100·title_match + 1·unique_token_match, Top-Fenster
# bis max_chars gesammelt.

def test_title_match_beats_token_only():
    """Fenster mit Exact-Title-Match muss Token-Spam schlagen (TOC-Bias-Fix).

    Beweis dass das Ranking arbeitet: Body-Fenster (Title-Match +100) drin,
    TOC-Fenster (nur Tokens) raus.
    """
    toc = "TOC_MARKER agent system multi pattern guardrails introduction overview"
    body = "multi-agent system is a coordinated set of agents working together"
    pad = " ".join(["filler"] * 500)
    text = f"{toc} {pad} {body} {pad}"
    out = concept_text_window(
        text,
        ["Multi-Agent System", "agent", "system", "multi"],
        window_words=200, max_chars=1500,
    )
    assert "multi-agent system is a coordinated" in out
    assert "TOC_MARKER" not in out  # TOC-Fenster wurde verworfen


def test_unique_token_count_not_repetition():
    """Wiederholtes Generic-Token schlägt nicht ein Fenster mit mehr unique Tokens."""
    # Fenster A: 50x "agent", 0x andere Tokens → score = 1
    a = " ".join(["agent"] * 50)
    pad = " ".join(["filler"] * 500)
    # Fenster B: je 1x "agent", "system", "multi" → score = 3
    b = "agent system multi context"
    text = f"{a} {pad} {b} {pad}"
    out = concept_text_window(
        text,
        ["Foo", "agent", "system", "multi"],  # kein Title-Match möglich
        window_words=200, max_chars=400,
    )
    assert "agent system multi context" in out


def test_no_match_returns_empty():
    text = "nothing related here at all"
    assert concept_text_window(text, ["target", "xyz"], window_words=50) == ""


def test_empty_search_terms_returns_prefix():
    text = "abc def ghi jkl"
    out = concept_text_window(text, [], max_chars=10)
    assert out == text[:10]


def test_document_order_preserved():
    """Selected Top-Fenster werden in Dokumentenreihenfolge gemerged."""
    early = "TARGET something here at start"
    late = "TARGET another instance late in doc"
    pad = " ".join(["filler"] * 600)  # genug damit Fenster getrennt sind
    text = f"{early} {pad} {late}"
    out = concept_text_window(
        text, ["TARGET"], window_words=200, max_chars=8000,
    )
    pos_early = out.find("at start")
    pos_late = out.find("late in doc")
    assert pos_early >= 0 and pos_late >= 0
    assert pos_early < pos_late


def test_respects_max_chars_budget():
    """max_chars wird respektiert — kleinere Fenster zwingen echtes Budget-Auffüllen.

    Mit window_words=50 ist ein Single-Chunk ~300 chars; bei max_chars=500
    werden mehrere Chunks selektiert und der Loop bricht beim Limit ab
    (statt durch `and picked`-Bypass beim ersten Chunk).
    """
    text = " ".join(["TARGET"] * 1000)
    out = concept_text_window(text, ["TARGET"], window_words=50, max_chars=500)
    assert len(out) <= 600  # Toleranz für Trenner-Overhead, nicht für komplettes Bypass


def test_single_chunk_oversize_still_returned():
    """`and picked`-Bypass: erster Chunk wird auch zurückgegeben wenn er allein
    schon das Budget sprengt — leerer String wäre für Downstream nutzlos."""
    text = " ".join(["TARGET"] * 500)
    out = concept_text_window(text, ["TARGET"], window_words=400, max_chars=100)
    assert len(out) > 100  # bewusster Bypass
    assert "TARGET" in out
