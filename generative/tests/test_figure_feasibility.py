"""Tests fuer die deterministische Figur-Feasibility-Probe.

Die Probe ist absichtlich LLM-frei und bleibt ein entkoppeltes Eval-Werkzeug:
Caption-Matching, Seitennaehe zu Chunks und PyMuPDF-Signale werden separat
bewertet, ohne den Orchestrator oder den Vault-Writer zu beruehren.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.pdf_chunker import Chunk

from eval_figure_feasibility import (
    PageVisualSignals,
    chunk_summary_rows,
    classify_page_signals,
    find_figure_captions,
    match_caption_to_chunk,
    report_has_no_hardcoded_literature_paths,
)


def test_find_figure_captions_matches_german_and_english_labels():
    text = (
        "Abbildung 1: Modell der Informationssuche.\n"
        "Fliesstext.\n"
        "Fig. 2. Search process diagram.\n"
        "Figure 3: Behaviour over time."
    )

    captions = find_figure_captions(text, page=7)

    assert [c.label for c in captions] == ["Abbildung 1", "Fig. 2", "Figure 3"]
    assert captions[0].page == 7
    assert captions[0].text == "Abbildung 1: Modell der Informationssuche."


def test_find_figure_captions_ignores_table_captions_and_table_context():
    text = (
        "Tabelle 1: Ergebnisse der Befragung.\n"
        "Table 2. Performance overview.\n"
        "Table 3: Abb. 2 in the appendix is referenced here.\n"
        "Die Abbildung der Stichprobe ist kein nummerierter Caption-Block."
    )

    captions = find_figure_captions(text, page=3)

    assert captions == []


def test_find_figure_captions_ignores_running_text_references():
    text = "Abbildung 1 zeigt, dass die Verteilung der Level ungleich ist."

    captions = find_figure_captions(text, page=9)

    assert captions == []


def test_match_caption_to_chunk_uses_page_range_inclusively():
    chunks = [
        Chunk("Abschnitt 1", "text", 0, page_start=1, page_end=2),
        Chunk("Abschnitt 2", "text", 1, page_start=3, page_end=5),
    ]
    caption = find_figure_captions("Figure 4: Information seeking stages.", page=5)[0]

    match = match_caption_to_chunk(caption, chunks)

    assert match is not None
    assert match.chunk_title == "Abschnitt 2"
    assert match.chunk_index == 1
    assert match.reason == "page_range"


def test_match_caption_to_chunk_returns_none_when_no_page_range_matches():
    chunks = [Chunk("Abschnitt 1", "text", 0, page_start=1, page_end=2)]
    caption = find_figure_captions("Abb. 9: Spaeteres Diagramm.", page=4)[0]

    assert match_caption_to_chunk(caption, chunks) is None


def test_classify_page_signals_distinguishes_raster_vector_caption_and_empty():
    assert classify_page_signals(PageVisualSignals(page=1, raster_images=2, vector_drawings=0, captions=[])) == "raster"
    assert classify_page_signals(PageVisualSignals(page=2, raster_images=0, vector_drawings=5, captions=[])) == "vector_or_composite"
    assert classify_page_signals(PageVisualSignals(page=3, raster_images=1, vector_drawings=5, captions=[])) == "vector_or_composite"
    captions = find_figure_captions("Figure 1: Conceptual model.", page=4)
    assert classify_page_signals(PageVisualSignals(page=4, raster_images=0, vector_drawings=0, captions=captions)) == "caption_only/no_visual_signal"
    assert classify_page_signals(PageVisualSignals(page=5, raster_images=0, vector_drawings=0, captions=[])) == "no_signal"


def test_report_has_no_hardcoded_literature_paths():
    assert report_has_no_hardcoded_literature_paths()


def test_chunk_summary_rows_exclude_chunk_text_from_report_payload():
    chunks = [Chunk("Abschnitt 1", "sensitive full chunk text", 0, page_start=1, page_end=3)]

    rows = chunk_summary_rows(chunks)

    assert rows == [{"title": "Abschnitt 1", "index": 0, "page_start": 1, "page_end": 3}]
    assert "sensitive full chunk text" not in str(rows)
