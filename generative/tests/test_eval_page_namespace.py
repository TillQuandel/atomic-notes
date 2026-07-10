"""Tests fuer #80 Fund 1: Stage-8-Eval nutzte den physischen PDF-Seitenindex statt
der Druckseiten-Labels aus `/PageLabels` (seit PR #79 der Namespace von
`source_anchors`, siehe pdf_chunker.pdf_to_pages). Divergenz nur bei label-
tragenden PDFs — PDFs ohne Labels bleiben bit-identisch (i+1-Fallback).

Betroffene Konsumenten:
- eval_quality_v2._pdf_sentences/build_chunks -> Chunk.pages -> best_page
  (persistiert in quality_history.jsonl, Judge-Pool-Header "[K1] Seiten {pages}"
  in eval_quality_v4.py, das dieselben Funktionen importiert).
- eval_quality_v4._pdf_artifacts (Stage-8-PDF-Memoisierung, #151) — derselbe Fix
  muss hier mitziehen, da sie _pdf_sentences direkt aufruft.
- eval_quality.eval_note (v1): source_anchors sind seit PR #79 Druckseiten-Labels,
  wurden aber weiter als physischer pdf_doc-Index verwendet.

Die committed PageLabels-Fixture (fixtures/pagelabels_arabic.pdf, #154) traegt nur
kurze CONTENT-Marker (<20 Zeichen) -- zu kurz fuer _pdf_sentences' Mindestlaenge
von 20 Zeichen pro Satz. Fuer Satz-/Chunk-Ebenen-Tests wird hier deshalb eine
eigene Fixture mit demselben Zwei-Stufen-Rezept gebaut (_build_text_pdf +
_with_page_labels aus fixtures_gen/make_pagelabel_fixtures.py — echter pypdf-
Trailer-Roundtrip, kein Mock von `_pdf_page_labels`), nur mit laengeren Saetzen.
Die pdf_chunker-Ebene (anchor_page_numbers/physical_pages_by_anchor) ist bereits
direkt gegen die committed Fixture abgesichert (test_pagelabels_fixture.py).
"""

from __future__ import annotations

from pathlib import Path

from generative.eval_quality_v2 import build_chunks
from generative.tests.fixtures_gen.make_pagelabel_fixtures import _build_text_pdf, _with_page_labels

_SENTENCES = [
    "Wilson beschreibt Informationsverhalten als ein uebergeordnetes Rahmenkonzept.",
    "Das ISP Modell von Kuhlthau umfasst sechs aufeinanderfolgende Phasen genau.",
    "Berrypicking betont die iterative Natur realer Suchprozesse sehr deutlich.",
    "Serendipity spielt bei der Entdeckung unerwarteter Quellen eine grosse Rolle.",
]


def _make_labeled_pdf(tmp_path: Path, pages_text: list[str], start_label: int, name: str = "labeled.pdf") -> Path:
    raw = _build_text_pdf(pages_text)
    labeled = _with_page_labels(
        raw, [dict(page_index_from=0, page_index_to=len(pages_text) - 1, style="/D", start=start_label)]
    )
    out_path = tmp_path / name
    out_path.write_bytes(labeled)
    return out_path


def _make_plain_pdf(tmp_path: Path, pages_text: list[str], name: str = "plain.pdf") -> Path:
    out_path = tmp_path / name
    out_path.write_bytes(_build_text_pdf(pages_text))
    return out_path


# ---- eval_quality_v2.build_chunks: Chunk.pages im Label-Namespace ----------


def test_build_chunks_carries_print_labels_not_physical_index(tmp_path):
    pdf_path = _make_labeled_pdf(tmp_path, _SENTENCES, start_label=159)

    chunks = build_chunks(pdf_path)

    all_pages = {p for c in chunks for p in c.pages}
    assert all_pages == {159, 160, 161, 162}


def test_build_chunks_no_labels_regression_stays_physical_index(tmp_path):
    """Regressionstest: PDF ohne /PageLabels -> Seitenzahlen unveraendert i+1."""
    pdf_path = _make_plain_pdf(tmp_path, _SENTENCES)

    chunks = build_chunks(pdf_path)

    all_pages = {p for c in chunks for p in c.pages}
    assert all_pages == {1, 2, 3, 4}


# ---- eval_quality_v4._pdf_artifacts: dieselbe Korrektur fuer Stage 8 -------


def test_pdf_artifacts_chunks_carry_print_labels(tmp_path):
    from generative import eval_quality_v4 as eq4

    eq4._reset_pdf_caches()
    pdf_path = _make_labeled_pdf(tmp_path, _SENTENCES, start_label=159, name="v4.pdf")

    chunks = eq4._pdf_artifacts(pdf_path).chunks

    all_pages = {p for c in chunks for p in c.pages}
    assert all_pages == {159, 160, 161, 162}


# ---- eval_quality.eval_note (v1): Anker-Seite vor pdf_doc-Zugriff aufloesen --


def test_eval_note_v1_uses_physical_page_for_label_anchor(tmp_path, monkeypatch):
    """Weisse-Box-Beweis: welche physischen Seiten `_extract_page_text` sieht, wenn
    der Note-Anker eine Druckseiten-Label-Zahl traegt (S. 160 = physische Seite 2,
    da start_label=159 -> Seite 1 = Label 159). calls muss [1, 2, 3] sein: Seite 1
    ist der feste Sprachdetektions-Sample (immer physisch 1), 2/3 sind physische
    Anker- + Folgeseite -- NICHT [1, 160, 161] (Label direkt als physischer Index
    missbraucht) und NICHT [1, 2, 161] (Folgeseite ueber Label+1 statt physisch+1,
    'Labels koennen springen')."""
    from generative import eval_quality as eq1

    pdf_path = _make_labeled_pdf(tmp_path, _SENTENCES, start_label=159)
    note_path = tmp_path / "note.md"
    note_path.write_text(
        "# Test\n\nDas ISP Modell von Kuhlthau umfasst sechs aufeinanderfolgende Phasen genau[^1].\n\n"
        "[^1]: quelle.pdf, S. 160.\n",
        encoding="utf-8",
    )

    calls: list[int] = []
    real_extract = eq1._extract_page_text

    def spy(pdf_doc, page_num):
        calls.append(page_num)
        return real_extract(pdf_doc, page_num)

    monkeypatch.setattr(eq1, "_extract_page_text", spy)

    result = eq1.eval_note(note_path, pdf_path, "v-test")

    assert calls == [1, 2, 3]
    assert result["anchors_not_parseable"] == 0
    assert result["anchors_confirmed"] == 1


def test_eval_note_v1_regression_no_labels_stays_physical_index(tmp_path, monkeypatch):
    """Regressionstest: PDF ohne /PageLabels -> Anker-Seite == physischer Index,
    unveraendertes Verhalten."""
    from generative import eval_quality as eq1

    pdf_path = _make_plain_pdf(tmp_path, _SENTENCES, name="plain_v1.pdf")
    note_path = tmp_path / "note.md"
    note_path.write_text(
        "# Test\n\nDas ISP Modell von Kuhlthau umfasst sechs aufeinanderfolgende Phasen genau[^1].\n\n"
        "[^1]: quelle.pdf, S. 2.\n",
        encoding="utf-8",
    )

    calls: list[int] = []
    real_extract = eq1._extract_page_text

    def spy(pdf_doc, page_num):
        calls.append(page_num)
        return real_extract(pdf_doc, page_num)

    monkeypatch.setattr(eq1, "_extract_page_text", spy)

    result = eq1.eval_note(note_path, pdf_path, "v-test")

    assert calls == [1, 2, 3]
    assert result["anchors_confirmed"] == 1


def _make_gap_labeled_pdf(tmp_path: Path, pages_text: list[str], name: str = "gap_labeled.pdf") -> Path:
    """PDF mit NICHT-kontiguen (aber monotonen -> usable) Druckseiten-Labels:
    Seiten 1-2 tragen 159,160; Seiten 3-4 springen auf 200,201 (Faltseiten-/
    Errata-Fall aus dem physisch+1-Kommentar in eval_quality.eval_note)."""
    raw = _build_text_pdf(pages_text)
    labeled = _with_page_labels(
        raw,
        [
            dict(page_index_from=0, page_index_to=1, style="/D", start=159),
            dict(page_index_from=2, page_index_to=len(pages_text) - 1, style="/D", start=200),
        ],
    )
    out_path = tmp_path / name
    out_path.write_bytes(labeled)
    return out_path


def test_gap_labels_follow_page_is_physical_not_label_plus_one(tmp_path, monkeypatch):
    """Codex-Review-Wunsch: Beweis der physisch+1-Entscheidung an springenden
    Labels (159,160,200,201). Anker 'S. 160' = physische Seite 2; die Folgeseite
    ist physisch 3 (= Label 200) -- Label+1 (161) existiert gar nicht. calls muss
    [1, 2, 3] sein, und das Mapping muss die Luecke exakt abbilden."""
    from generative import eval_quality as eq1
    from generative.pipeline.pdf_chunker import anchor_page_numbers, physical_pages_by_anchor

    pdf_path = _make_gap_labeled_pdf(tmp_path, _SENTENCES)

    assert anchor_page_numbers(pdf_path, len(_SENTENCES)) == [159, 160, 200, 201]
    assert physical_pages_by_anchor(pdf_path, len(_SENTENCES)) == {159: 1, 160: 2, 200: 3, 201: 4}

    note_path = tmp_path / "note.md"
    note_path.write_text(
        "# Test\n\nDas ISP Modell von Kuhlthau umfasst sechs aufeinanderfolgende Phasen genau[^1].\n\n"
        "[^1]: quelle.pdf, S. 160.\n",
        encoding="utf-8",
    )

    calls: list[int] = []
    real_extract = eq1._extract_page_text

    def spy(pdf_doc, page_num):
        calls.append(page_num)
        return real_extract(pdf_doc, page_num)

    monkeypatch.setattr(eq1, "_extract_page_text", spy)

    result = eq1.eval_note(note_path, pdf_path, "v-test")

    assert calls == [1, 2, 3]
    assert result["anchors_confirmed"] == 1
