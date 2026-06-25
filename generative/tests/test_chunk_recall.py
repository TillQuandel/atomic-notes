"""Tests für eval_chunk_recall — deterministische Phase-A-Mechanik-Messung.

Phase A misst Retrieval-Eigenschaften der deterministischen Chunking-/Overview-
Funktionen OHNE LLM-Call. Zwei Metriken:
  1. straddle_stats  — Sätze die von Wort-Split-Chunk-Grenzen zerschnitten werden
                       (Overlap-Chunking-Nutzen).
  2. overview_coverage — welche Konzepte ihre Titel-Token-Evidenz im
                         extract_overview()-Output behalten (Planner-Recall-Decke).
"""
from __future__ import annotations


import pytest

from generative.eval_chunk_recall import straddle_stats, overview_coverage, _label_concept


# ---- straddle_stats: Boundary-Loss ohne Overlap ----------------------------

def test_no_straddle_when_chunk_holds_whole_sentence():
    text = "alpha beta gamma."
    stats = straddle_stats(text, size=10, overlap=0)
    assert stats["n_sentences"] == 1
    assert stats["n_straddling"] == 0


def test_sentence_cut_by_boundary_counts_as_straddling():
    # words: [a b c. d e f.] idx 0..5; size=4,overlap=0 → chunks [0:4],[4:6]
    # Satz2 = idx 3,4,5: idx3 in chunk0, idx4-5 in chunk1 → in keinem ganz → straddle
    text = "a b c. d e f."
    stats = straddle_stats(text, size=4, overlap=0)
    assert stats["n_sentences"] == 2
    assert stats["n_straddling"] == 1


def test_overlap_recovers_straddling_sentence():
    # stride = size-overlap = 2; chunk [2:6] enthält Satz2 (idx 3,4,5) vollständig
    text = "a b c. d e f."
    stats = straddle_stats(text, size=4, overlap=2)
    assert stats["n_straddling"] == 0


def test_page_markers_do_not_create_pseudo_sentences():
    # `[S. 12]` endet auf `.` im Token `[S.` → würde ohne Strip einen Pseudo-Satz erzeugen
    # und damit Boundary-Loss verfälschen (Codex-Finding 2026-06-04).
    text = "alpha beta gamma. [S. 12] delta epsilon zeta."
    stats = straddle_stats(text, size=100, overlap=0)
    assert stats["n_sentences"] == 2


def test_straddle_stats_reports_word_split_mode():
    # Klarstellen, dass die Metrik den _split_by_words-Pfad misst, nicht den Default.
    stats = straddle_stats("a b c.", size=10)
    assert stats["mode"] == "word_split"


@pytest.mark.parametrize("size,overlap", [(0, 0), (4, 4), (4, 5), (4, -1)])
def test_straddle_stats_rejects_invalid_params(size, overlap):
    # Guard gegen degenerierte Messungen: size>0 und 0<=overlap<size.
    with pytest.raises(ValueError):
        straddle_stats("a b c.", size=size, overlap=overlap)


# ---- overview_coverage: Planner-Recall-Decke -------------------------------

def test_concept_in_intro_covered_at_all_thresholds():
    # Konzept-Tokens stehen ganz am Anfang → immer im Intro-Budget von extract_overview
    text = "Zebra Quantum " + "lorem " * 2000
    res = overview_coverage(text, ["Zebra Quantum"])
    assert res["covered"][0.5] == 1
    assert res["covered"][1.0] == 1
    assert res["exact_phrase"] == 1


def test_concept_buried_deep_in_chapter_missed_at_all_thresholds():
    # Eindeutiges Konzept tief in Kapitel 2 — jenseits von Intro, Kapitel-Snippet
    # und Outro-Budget → extract_overview sieht es nicht, obwohl es im Volltext steht.
    filler = "lorem " * 1000
    text = (
        "1 Aaaa\n" + "lorem " * 50 + "\n"
        "2 Bbbb\n" + filler + "Schattenkonzept Tiefgraben " + filler + "\n"
        "3 Cccc\n" + "lorem " * 50 + "\n"
        "4 Dddd\n" + "lorem " * 50 + "\n"
        "5 Eeee\n" + "lorem " * 50
    )
    res = overview_coverage(text, ["Schattenkonzept Tiefgraben"])
    assert res["covered"][0.5] == 0
    assert res["n_in_fulltext"] == 1
    assert "Schattenkonzept Tiefgraben" in res["missed_strict"]


def test_partial_token_overlap_is_threshold_artifact():
    # 'alpha' im Intro, 'betaunique' tief vergraben → cov_ov = 1/2 = 0.5.
    # Konzept "alpha betaunique" gilt bei min_coverage=0.5 als covered,
    # bei 1.0 NICHT — genau das Schwellen-Artefakt aus dem Codex-Review.
    filler = "lorem " * 1000
    text = (
        "alpha 1 Aaaa\n" + "lorem " * 50 + "\n"
        "2 Bbbb\n" + filler + "betaunique " + filler + "\n"
        "3 Cccc\n" + "lorem " * 50 + "\n"
        "4 Dddd\n" + "lorem " * 50 + "\n"
        "5 Eeee\n" + "lorem " * 50
    )
    res = overview_coverage(text, ["alpha betaunique"])
    assert res["n_in_fulltext"] == 1
    assert res["covered"][0.5] == 1
    assert res["covered"][1.0] == 0


# ---- _label_concept: Quelle→Konzept-Referenz aus Kalibrierungs-Notes -------

def test_label_concept_parses_pdf_and_title():
    text = (
        "# Label 06/30 [HYBRID] — vault__10 Gebote der Frageformulierung.md\n"
        "\n"
        "- **PDF**: `C:/x/Porst - 2014 - Fragebogen Ein Arbeitsbuch.pdf`\n"
        "- **Claims total**: 10\n"
    )
    pdf, title = _label_concept(text)
    assert title == "10 Gebote der Frageformulierung"
    assert "Porst - 2014" in pdf


def test_label_concept_strips_inbox_prefix():
    text = (
        "# Label 16/30 [HYBRID] — inbox__Geschichte der Information-Behavior-Forschung.md\n"
        "- **PDF**: `C:/x/Bates - 2017 - Information Behavior.pdf`\n"
    )
    pdf, title = _label_concept(text)
    assert title == "Geschichte der Information-Behavior-Forschung"
    assert "Bates - 2017" in pdf
