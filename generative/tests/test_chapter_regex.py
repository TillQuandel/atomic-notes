"""Tests für die erweiterte _CHAPTER_RE in pdf_chunker.

Erkennt zusätzlich zu arabischen Ziffern: römische, ausgeschriebene Zahlen, neue
Prefixe (Beitrag, Section), längere Titel bis 120 Zeichen.

Dual-Use-Schutz: TOC-Trail-Filter verhindert dass Inhaltsverzeichnis-Zeilen oder
Aufzählungs-Bulletpoints als Kapitel-Heading durchgehen — sowohl in
split_by_chapters() als auch in drop_frontmatter_pages().
"""
from __future__ import annotations
import sys
from pathlib import Path


from generative.pipeline.pdf_chunker import (
    _CHAPTER_RE,
    _is_real_chapter_match,
    split_by_chapters,
    drop_frontmatter_pages,
)


def _matches(text: str) -> list[str]:
    return [m.group(0).strip() for m in _CHAPTER_RE.finditer(text) if _is_real_chapter_match(m)]


# ---- Positiv: bestehende Pattern bleiben grün ----------------------------

def test_arabic_chapter_still_matches():
    text = "[S. 5]\n1 Einleitung\nText folgt hier.\n[S. 12]\n2 Methoden\nMehr Text."
    out = _matches(text)
    assert len(out) == 2


def test_arabic_with_subsection():
    text = "[S. 1]\n2.1 Untersuchungsdesign\nText.\n[S. 7]\n2.2 Stichprobe\nText."
    out = _matches(text)
    assert len(out) == 2


def test_chapter_prefix_still_matches():
    text = "Kapitel 2 Grundlagen\nText.\nChapter 3 Results\nText."
    out = _matches(text)
    assert len(out) == 2


# ---- Positiv: neue Pattern -----------------------------------------------

def test_roman_numeral_chapter():
    text = "I. Einleitung\nText hier.\nII. Theorie\nMehr Text."
    out = _matches(text)
    assert len(out) == 2, f"Expected 2 roman chapters, got {out!r}"


def test_beitrag_prefix():
    text = "Beitrag 3 Wissensorganisation in Bibliotheken\nText.\nBeitrag 4 Klassifikationssysteme\nMehr."
    out = _matches(text)
    assert len(out) == 2, f"Expected 2 Beitrag matches, got {out!r}"


def test_kapitel_zwei_ausgeschrieben():
    text = "Kapitel zwei Grundlagen der Theorie\nText.\nKapitel drei Methoden\nText."
    out = _matches(text)
    assert len(out) == 2, f"Expected 2 spelled-out matches, got {out!r}"


def test_long_title_up_to_120_chars():
    # Titel >60 (alte Grenze) aber <120 (neue Grenze)
    long_title = "Information Behavior in the Context of Digital Libraries and Academic Research Practices"
    text = f"3 {long_title}\nText folgt.\n4 Methoden\nMehr."
    out = _matches(text)
    assert len(out) == 2, f"Expected long title to match, got {out!r}"


# ---- Negativ: False-Positives müssen gefiltert werden --------------------

def test_toc_trail_with_dots_filtered():
    # Klassischer Inhaltsverzeichnis-Eintrag — KEIN echtes Kapitel-Heading
    text = "Inhalt\nI. Einleitung .................. 12\nII. Theorie ................... 24"
    out = _matches(text)
    assert out == [], f"TOC-Zeilen mit Punkten dürfen nicht matchen, got {out!r}"


def test_toc_trail_with_spaces_filtered():
    text = "Inhalt\n1 Einleitung                 12\n2 Methoden                  24"
    out = _matches(text)
    assert out == [], f"TOC-Zeilen mit Spacing+Seite dürfen nicht matchen, got {out!r}"


# ---- Integration: split_by_chapters --------------------------------------

def test_split_by_chapters_with_roman():
    text = (
        "[S. 1]\nI. Einleitung\nLanger Einleitungstext hier mit vielen Wörtern und Sätzen.\n"
        "[S. 10]\nII. Theorie\nTheoretischer Hintergrund wird ausgeführt mit Details.\n"
        "[S. 20]\nIII. Methoden\nMethodischer Ansatz folgt hier mit Erläuterungen.\n"
    )
    chunks = split_by_chapters(text)
    titles = [c.title for c in chunks]
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}: {titles}"
    assert any("Einleitung" in t for t in titles)


# ---- Integration: drop_frontmatter_pages mit römisch ---------------------

def test_drop_frontmatter_with_roman_chapter():
    pages = [
        (1, "Advance Praise for the book\nReviewer comments here."),
        (2, "Copyright 2020. All rights reserved.\nISBN 978-1-2345-6789-0"),
        (3, "Preface\nThis book covers important topics."),
        (4, "I. Einleitung\nDer eigentliche Inhalt beginnt hier."),
        (5, "Mehr Body-Inhalt."),
        (6, "II. Theorie\nWeiterer Inhalt."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 3, f"Expected 3 frontmatter pages dropped, got {dropped}"
    assert out[0][0] == 4
    assert "Einleitung" in out[0][1]


def test_drop_frontmatter_skips_roman_vorwort_as_chapter():
    # „I. Vorwort" matched die Regex, ist aber NICHT Body — Titel-Group enthält
    # eine Frontmatter-Phrase. Pipeline soll bis zum echten ersten Kapitel weitercroppen.
    pages = [
        (1, "Advance Praise\nReviewer A says nice things."),
        (2, "Copyright 2020.\nISBN 978-1-2345-6789-0"),
        (3, "I. Vorwort\nDanksagung an Kollegen und Familie für die Unterstützung."),
        (4, "II. Einleitung\nDer eigentliche Inhalt beginnt hier mit Theorie."),
        (5, "Mehr Body-Inhalt mit Substanz und vielen Details."),
        (6, "Noch mehr Body-Inhalt für die ratio-Berechnung."),
        (7, "Weiterer Body-Abschnitt mit Methodik."),
        (8, "Body-Kapitel zur Ergebnisdarstellung."),
        (9, "Body-Inhalt zum Diskussionsteil."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 3, f"Vorwort darf nicht als Body zählen, expected drop=3, got {dropped}"
    assert out[0][0] == 4
    assert "Einleitung" in out[0][1]


def test_drop_frontmatter_ignores_toc_match():
    # TOC-Seite enthält "I. Einleitung ........ 12" — das darf NICHT als erstes
    # Chapter zählen, sonst würde Frontmatter-Crop zu früh starten und das
    # echte erste Kapitel (Page 5) wäre nicht das first_chapter_idx.
    # 4 Frontmatter-Pages + 5 Body-Pages → drop_ratio 4/9 < 0.5 (kein Cap-Hit)
    pages = [
        (1, "Advance Praise\nReviewer A: great book."),
        (2, "Copyright 2020.\nISBN 978-..."),
        (3, "Inhalt\nI. Einleitung .................. 5\nII. Methoden ............... 20"),
        (4, "Preface\nBookmark text here."),
        (5, "I. Einleitung\nDer echte Inhalt beginnt jetzt."),
        (6, "Mehr Body-Inhalt mit Substanz und ausführlichen Erläuterungen."),
        (7, "Noch mehr Body-Inhalt mit Theorie und Beispielen aus der Praxis."),
        (8, "Weiterer Abschnitt mit Diskussion und Implikationen für die Forschung."),
        (9, "Abschluss-Kapitel mit Fazit und Ausblick auf zukünftige Arbeiten."),
    ]
    out, dropped = drop_frontmatter_pages(pages)
    assert dropped == 4, f"TOC darf nicht zu Page 3 als first_chapter führen, got dropped={dropped}"
    assert out[0][0] == 5
