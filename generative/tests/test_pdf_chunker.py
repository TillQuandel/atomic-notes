"""Tests für pdf_chunker — Frontmatter-Strip + Page-Range-Helpers.

Frontmatter-Strip verhindert dass `concept_text_window` Cluster im PDF-Frontmatter
(Advance Praise, Acknowledgments, Copyright, …) findet, wenn Konzept-Begriffe
dort en passant vorkommen — Hiatt 2026-05-10 ADKAR-Eval-Bug. Siehe
[[Atomic-Agent-Pipeline]] v24.
"""
from __future__ import annotations
import re


from generative.pipeline.pdf_chunker import drop_frontmatter_pages, page_range_of_text, concept_text_window


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


# ---- concept_text_window: Seiten-Marker-Reinjektion (#4 Anker-Clustering) ----
# Bug: concept_text_window klebt Top-Fenster aus dem ganzen Dokument zusammen.
# Beginnt ein Fenster mitten auf einer Seite, fehlt ihm der [S. N]-Marker (der
# stand am Seitenanfang, vor dem Fenster). Downstream (Extractor-LLM, Verifier,
# Renderer) leitet die Seite über "letzter [S. N]-Marker vor der Fundstelle" ab
# und erbt dann die Seite eines früheren Snippets → falsche Fußnoten-Seite.
# Merrill-Run 2026-06-24: Integration-Detail (echt S.8) bekam pauschal "S.3".

def _page_before(text: str, needle: str) -> str | None:
    """Letzter [S. N]-Marker vor `needle` — exakt wie Downstream die Seite ableitet."""
    pos = text.find(needle)
    if pos < 0:
        return None
    last = None
    for m in re.finditer(r"\[S\.\s*(\d+)\]", text):
        if m.start() > pos:
            break
        last = m.group(1)
    return last


def test_snippet_retains_correct_page_marker_when_window_starts_mid_page():
    """Ein Fenster, das mitten auf einer Seite beginnt, muss seinen korrekten
    [S. N]-Marker tragen — sonst erbt die Seitenableitung den Marker eines
    früheren Snippets (der reproduzierte #4-Bug). Marker line-isoliert wie aus
    pages_to_marked_text (`\\n\\n[S. N]\\n\\n`)."""
    pad = " ".join(["filler"] * 500)
    # S.1: Overview-Fenster mit Titel+Tokens (rankt hoch, trägt eigenen Marker)
    overview = "\n\n[S. 1]\n\nTARGET alpha beta gamma " + pad
    # S.2–S.4: Marker vorhanden, aber kein Token → nicht selektiert
    mid = "\n\n[S. 2]\n\n" + pad + "\n\n[S. 3]\n\n" + pad + "\n\n[S. 4]\n\n" + pad
    # S.5: Marker, dann >window_words filler, dann der Detail-Cluster → das
    # selektierte Detail-Fenster beginnt NACH dem [S. 5]-Marker.
    detail = "\n\n[S. 5]\n\n" + " ".join(["filler"] * 550) + " TARGET alpha beta delta DETAILNEEDLE"
    text = f"{overview}{mid}{detail}"

    out = concept_text_window(
        text, ["TARGET", "alpha", "beta"], window_words=400, max_chars=8000
    )

    assert "DETAILNEEDLE" in out, "Detail-Fenster muss selektiert sein"
    assert _page_before(out, "DETAILNEEDLE") == "5", (
        "Detail-Snippet muss seinen eigenen Seitenmarker S.5 tragen, "
        "nicht den S.1 des früheren Snippets erben"
    )


def test_inline_page_ref_not_treated_as_page_start():
    """Inline-Quellenverweis „vgl. [S. 12]" im Fließtext darf NICHT als
    Seitenanfang gelten — ein folgendes markerloses Snippet erbt die echte
    Seite (S.7), nicht die inline zitierte (S.12). (Codex-Review #4 MED.)"""
    # Echter Seitenanfang S.7 (line-isoliert), dann ein Inline-Ref im Fließtext,
    # dann >window_words filler, dann der Detail-Cluster.
    text = (
        "\n\n[S. 7]\n\nvgl. dazu [S. 12] in der Literatur "
        + " ".join(["filler"] * 550)
        + " TARGET alpha beta DETAILNEEDLE"
    )

    out = concept_text_window(
        text, ["TARGET", "alpha", "beta"], window_words=400, max_chars=8000
    )

    assert "DETAILNEEDLE" in out, "Detail-Fenster muss selektiert sein"
    assert _page_before(out, "DETAILNEEDLE") == "7", (
        "markerloses Detail-Snippet muss die echte Seite S.7 erben, "
        "nicht die inline zitierte S.12"
    )


# ---- _resolve_page_numbers (Druckseiten-Labels statt Form-Feed-Index) -----

def test_resolve_page_numbers_uses_numeric_labels():
    """Echte (arabische) Druckseiten-Labels werden als Seitenzahl genutzt — nicht
    die Form-Feed-Position. Buch-Kapitel/-Extrakt: PDF-Seite 1 trägt Druckseite 159."""
    from generative.pipeline.pdf_chunker import _resolve_page_numbers
    assert _resolve_page_numbers(["a", "b", "c"], ["159", "160", "161"]) == [
        (159, "a"), (160, "b"), (161, "c")
    ]


def test_resolve_page_numbers_roman_falls_back_to_index():
    """Nicht-numerische Labels (römisches Frontmatter) dürfen die \\d+-Anker-Kette
    nicht brechen → Fallback auf 1-basierten Form-Feed-Index."""
    from generative.pipeline.pdf_chunker import _resolve_page_numbers
    assert _resolve_page_numbers(["a", "b", "c"], ["xi", "xii", "1"]) == [
        (1, "a"), (2, "b"), (1, "c")
    ]


def test_resolve_page_numbers_no_labels_is_index():
    """Kein PageLabels-Eintrag (labels=None) → exakt das alte Verhalten (i+1)."""
    from generative.pipeline.pdf_chunker import _resolve_page_numbers
    assert _resolve_page_numbers(["a", "b"], None) == [(1, "a"), (2, "b")]


def test_resolve_page_numbers_length_mismatch_safe():
    """pdftotext kann eine Extraseite liefern (finaler \\f) → überzählige Seiten
    fallen sauber auf den Index zurück, kein IndexError."""
    from generative.pipeline.pdf_chunker import _resolve_page_numbers
    assert _resolve_page_numbers(["a", "b", "c"], ["159", "160"]) == [
        (159, "a"), (160, "b"), (3, "c")
    ]


def test_resolve_page_numbers_strips_whitespace_and_coerces_nonstr():
    """pypdf-Labels können Whitespace (' 159 ') oder (selten) non-str tragen →
    robust strippen/coercen statt aufs Form-Feed zurückzufallen oder zu crashen
    (Qwen-Review HIGH/MED, 2. Durchgang)."""
    from generative.pipeline.pdf_chunker import _resolve_page_numbers
    assert _resolve_page_numbers(["a", "b"], [" 159 ", 160]) == [(159, "a"), (160, "b")]


def test_usable_page_labels_gate_requires_numeric_and_unique():
    """Label-Modus nur bei vollständig numerischen UND eindeutigen Labels — sonst
    None. Verhindert Namespace-Kollision römisch↔arabisch (False-Bind in figure_alt)
    und mehrdeutige Index-Abbildung bei Duplikaten (Codex-Review, 2. Durchgang)."""
    from generative.pipeline.pdf_chunker import _usable_page_labels
    assert _usable_page_labels(["159", "160", "161"]) == ["159", "160", "161"]
    assert _usable_page_labels([" 159 ", "160"]) == [" 159 ", "160"]   # numerisch m. Whitespace ok
    assert _usable_page_labels(["xi", "xii", "1"]) is None             # gemischt römisch/arabisch
    assert _usable_page_labels(["1", "2", "2"]) is None                # doppelt → mehrdeutig
    assert _usable_page_labels(["100", "1", "2"]) is None              # nicht monoton → falsche Ranges
    # Zero-Padding-Duplikat: als Strings verschieden ("01"≠"1"), als Zahl gleich (1==1).
    # Muss abgelehnt werden, sonst zwei Seiten mit [S. 1] → False-Bind (Qwen-Review, 2026-06-27).
    assert _usable_page_labels(["01", "1", "2"]) is None
    # Unicode-Ziffern: str.isdigit() ist True für Superscripts (²), aber int("²") crasht.
    # isdecimal()-Gate lehnt sie ab statt sich auf das except im Aufrufer zu verlassen
    # (Codex-Review, 2026-06-27).
    assert _usable_page_labels(["1", "²"]) is None
    assert _usable_page_labels(None) is None
    assert _usable_page_labels([]) is None


# ---- assess_text_quality (G6/#27 — Textqualitäts-Gate) -------------------

def test_empty_text_is_empty_not_thin():
    from generative.pipeline.pdf_chunker import assess_text_quality
    q = assess_text_quality("")
    assert q.is_empty is True
    assert q.is_thin is False
    assert q.total_words == 0


def test_thin_scanned_text_flagged():
    # Gescanntes PDF ohne OCR: viele Seiten, kaum extrahierter Text (Rauschen).
    from generative.pipeline.pdf_chunker import assess_text_quality, pages_to_marked_text
    pages = [(n, "3 7") for n in range(1, 11)]  # 10 Seiten, je 2 "Wörter"
    text = pages_to_marked_text(pages)
    q = assess_text_quality(text)
    assert q.is_thin is True
    assert q.is_empty is False
    assert q.words_per_page < 50


def test_normal_dense_text_not_thin():
    from generative.pipeline.pdf_chunker import assess_text_quality, pages_to_marked_text
    body = " ".join(["wort"] * 300)  # 300 Wörter/Seite — normaler Fließtext
    pages = [(1, body), (2, body)]
    text = pages_to_marked_text(pages)
    q = assess_text_quality(text)
    assert q.is_thin is False
    assert q.is_empty is False
    assert q.words_per_page >= 50


def test_page_markers_not_counted_as_words():
    # Die `[S. N]`-Marker dürfen die Wortzahl nicht aufblähen (sonst sähe ein
    # leeres PDF mit 40 Seiten-Markern "wortreich" aus).
    from generative.pipeline.pdf_chunker import assess_text_quality, pages_to_marked_text
    pages = [(n, "") for n in range(1, 41)]  # 40 leere Seiten → nur Marker
    text = pages_to_marked_text(pages)
    q = assess_text_quality(text)
    assert q.total_words == 0
    assert q.is_empty is True


def test_inline_page_refs_not_counted_as_pages():
    # Codex-Review G6/#27: Inline-Quellenverweise "[S. 12]" im Fließtext (deutsche
    # Zitierweise) dürfen NICHT als eigene Seiten gezählt werden — sonst wird
    # words_per_page künstlich gedrückt und ein gutes kurzes PDF fälschlich als
    # is_thin (False-Positive-Warnung) markiert. Nur zeilen-isolierte Pipeline-Marker
    # zählen als echte Seiten.
    from generative.pipeline.pdf_chunker import assess_text_quality, pages_to_marked_text
    body = "Mustertext " * 100 + "vgl. dazu [S. 12] sowie [S. 34] in der Literatur."
    pages = [(1, body)]
    text = pages_to_marked_text(pages)
    q = assess_text_quality(text)
    assert q.pages == 1        # nur der eine echte Pipeline-Marker
    assert q.is_thin is False  # ~100 Wörter auf 1 Seite ist nicht dünn


# ---- pdf_metadata: Info-Dict-Autor/Jahr sind NICHT zitierfähig -----------
# Universelle Regel (nicht quellen-spezifisch): pdfinfo-`Author` (= Datei-
# Ersteller) und das Jahr aus `CreationDate` (= Speicher-/Abtipp-Zeitpunkt)
# identifizieren NICHT Werk-Autor bzw. Publikationsjahr. Sie dürfen nie als
# Zitier-Autor/-Jahr durchgereicht werden — sonst systematische Fehlattribution
# bei abgetippten/gescannten/neu-gespeicherten PDFs (realer Fall: ein in Word
# abgetipptes Knowles-Kapitel trug `Author: Pierre Landry` / CreationDate 2019
# → alle Notes zitierten "Landry 2019" statt Knowles).
from generative.pipeline.pdf_chunker import _parse_pdfinfo_output

_PDFINFO_RETYPED = (
    "Title:          What Is Andragogy?\n"
    "Author:         Pierre Landry\n"
    "Creator:        Microsoft Word 2016\n"
    "CreationDate:   Wed Mar 20 18:27:09 2019 CET\n"
    "Pages:          25\n"
)


def test_pdfinfo_author_not_exposed_as_citation_author():
    meta = _parse_pdfinfo_output(_PDFINFO_RETYPED)
    assert "Author" not in meta
    assert meta.get("InfoDictAuthor") == "Pierre Landry"


def test_pdfinfo_creationdate_not_exposed_as_citation_year():
    meta = _parse_pdfinfo_output(_PDFINFO_RETYPED)
    assert "Year" not in meta
    assert meta.get("InfoDictCreationYear") == "2019"


def test_pdfinfo_keeps_title_and_pages():
    meta = _parse_pdfinfo_output(_PDFINFO_RETYPED)
    assert meta.get("Title") == "What Is Andragogy?"
    assert meta.get("Pages") == "25"
