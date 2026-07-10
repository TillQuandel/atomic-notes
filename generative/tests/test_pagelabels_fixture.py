"""Tests gegen REALE `/PageLabels`-Fixtures (Issue #154).

Schließt den Suite-Blindspot: Bis hierher trug KEINE Fixture-PDF `/PageLabels`,
darum lief `_pdf_page_labels` (`pdf_chunker.py`, echtes pypdf-Trailer-Parsing)
in keinem Test gegen eine real gelabelte PDF — alle Tests übergaben Label-Listen
direkt oder patchten die Funktion weg. Damit war der #79-Wurzelfix (Druckseiten
aus `/PageLabels` + `_usable_page_labels`-Gate gegen römisch↔arabisch-Kollision)
ungeschützt.

Fixtures (committed, erzeugt von `fixtures_gen/make_pagelabel_fixtures.py`):
- pagelabels_arabic.pdf       : rein arabische Labels 159–162 → Happy-Path.
- pagelabels_roman_arabic.pdf : gemischt röm.(i,ii)+arab. → Gate → i+1-Fallback.

Prämissen-Korrektur ggü. Issue-Text: Das Issue skizziert EINE gemischte Fixture
(röm. Frontmatter + arabisch) für den Druckseiten-Pfad. Das ist mit dem
`_usable_page_labels`-Gate nicht möglich — eine Liste mit römischen Labels ist
nicht vollständig `isdecimal()` und wird komplett verworfen (→ i+1 für ALLE
Seiten). Die gemischte PDF ist daher exakt der Gate-Fall (#79-Codex-Fund), der
Druckseiten-Pfad braucht eine rein arabische Fixture. Beide Teile sind hier
getrennt abgedeckt.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from generative.pipeline.pdf_chunker import (
    _pdf_page_labels,
    anchor_page_numbers,
    physical_pages_by_anchor,
    pdf_to_pages,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARABIC = FIXTURES / "pagelabels_arabic.pdf"
ROMAN_ARABIC = FIXTURES / "pagelabels_roman_arabic.pdf"

# pdf_to_pages ruft pdftotext (poppler) im Subprozess. CI hat es auf allen 3 OS
# (ci.yml); lokal ohne poppler → Skip statt Hard-Fail (konsistent zu
# test_example_pdf.py / test_ci_smoke_e2e.py).
needs_pdftotext = pytest.mark.skipif(
    shutil.which("pdftotext") is None,
    reason="pdftotext (poppler) nicht auf PATH — in CI vorhanden.",
)


# ---- (a) echter pypdf-Roundtrip, KEIN Monkeypatch --------------------------


def test_pdf_page_labels_reads_real_arabic_labels():
    """`_pdf_page_labels` liest die echten arabischen Druckseiten aus dem PDF
    (pypdf-Trailer-Parsing, kein gestubbtes Signal)."""
    assert _pdf_page_labels(ARABIC) == ["159", "160", "161", "162"]


def test_pdf_page_labels_gate_rejects_mixed_roman_arabic():
    """Gemischt röm./arab. Fixture: `_usable_page_labels` verwirft die Liste →
    `_pdf_page_labels` gibt None. Der #79-Codex-Fund als Regression gegen eine
    ECHTE gelabelte PDF (nicht gegen eine direkt übergebene Liste)."""
    assert _pdf_page_labels(ROMAN_ARABIC) is None


# ---- (b) End-to-End: Druckseiten statt Form-Feed-Position -------------------


@needs_pdftotext
def test_pdf_to_pages_uses_print_labels_not_index():
    """Happy-Path end-to-end: physische Seite 1 trägt Druckseite 159 (nicht 1),
    physische Seite 3 trägt 161 (nicht 3). Beweist den `/PageLabels`→Druckseiten-
    Pfad durch die volle pdftotext-Kette."""
    pages = pdf_to_pages(ARABIC)
    nums = [n for n, _ in pages]
    assert nums == [159, 160, 161, 162]
    # Physische 3. Seite → Druckseite 161, und ihr Text ist eindeutig zuordenbar.
    page3_num, page3_text = pages[2]
    assert page3_num == 161
    assert "CONTENT-161" in page3_text


# ---- (c) Namespace-Gate: i+1-Fallback bei gemischten Labels -----------------


@needs_pdftotext
def test_pdf_to_pages_falls_back_to_index_on_mixed_labels():
    """Gemischt röm./arab.: das Gate erzwingt den i+1-Fallback für ALLE Seiten.
    Die physisch 3. Seite (erste arabische, Druckseite-Text `CONTENT-159`)
    bekommt Position 3 — NICHT 159. Genau die römisch↔arabisch-Namespace-
    Kollision, die #79 (Codex, 2. Durchgang) verhindern sollte."""
    pages = pdf_to_pages(ROMAN_ARABIC)
    nums = [n for n, _ in pages]
    assert nums == [1, 2, 3, 4, 5, 6]
    page3_num, page3_text = pages[2]
    assert page3_num == 3
    assert "CONTENT-159" in page3_text  # Inhalt ist Druckseite 159, Position ist 3


# ---- (d) anchor_page_numbers / physical_pages_by_anchor gegen die echte Fixture
#          (#80 Fund 1: gemeinsames Mapping fuer eval_quality.py/_v2.py/_v4.py) --


def test_anchor_page_numbers_matches_real_arabic_labels():
    """Kein pypdf-Mock: echter Trailer-Roundtrip liefert 159..162 je physischem
    0-basiertem Index — derselbe Namespace wie pdf_to_pages (a)."""
    assert anchor_page_numbers(ARABIC, 4) == [159, 160, 161, 162]


def test_anchor_page_numbers_falls_back_on_mixed_roman_arabic():
    """Gate-Fall: gemischte Labels -> _pdf_page_labels liefert None -> i+1 fuer
    ALLE Seiten, konsistent mit pdf_to_pages' Fallback (c)."""
    assert anchor_page_numbers(ROMAN_ARABIC, 6) == [1, 2, 3, 4, 5, 6]


def test_physical_pages_by_anchor_reverses_real_arabic_labels():
    """Umkehrung: Druckseiten-Label -> physischer 1-basierter Index, fuer den
    v1-Eval-Pfad (eval_quality.py), der Anker-Seiten vor dem pdf_doc-Zugriff
    zurueckuebersetzen muss."""
    assert physical_pages_by_anchor(ARABIC, 4) == {159: 1, 160: 2, 161: 3, 162: 4}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
