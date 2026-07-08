"""Tests fuer die deterministische Figur-Feasibility-Probe.

Die Probe ist absichtlich LLM-frei und bleibt ein entkoppeltes Eval-Werkzeug:
Caption-Matching, Seitennaehe zu Chunks und PyMuPDF-Signale werden separat
bewertet, ohne den Orchestrator oder den Vault-Writer zu beruehren.
"""

from __future__ import annotations

from pathlib import Path

from generative.pipeline.pdf_chunker import Chunk

from generative.eval_figure_feasibility import (
    PageVisualSignals,
    _page_visual_signals,
    chunk_summary_rows,
    classify_page_signals,
    find_figure_captions,
    match_caption_to_chunk,
    report_has_no_hardcoded_literature_paths,
    warning_for_classification,
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


def test_classify_page_signals_uses_only_caption_and_raster():
    caption = find_figure_captions("Figure 1: Conceptual model.", page=1)
    assert (
        classify_page_signals(PageVisualSignals(page=1, raster_images=2, vector_drawings=0, captions=caption))
        == "captioned_raster"
    )
    assert (
        classify_page_signals(PageVisualSignals(page=2, raster_images=0, vector_drawings=0, captions=caption))
        == "captioned_no_raster"
    )
    assert (
        classify_page_signals(PageVisualSignals(page=3, raster_images=2, vector_drawings=0, captions=[]))
        == "raster_uncaptioned"
    )
    assert (
        classify_page_signals(PageVisualSignals(page=4, raster_images=0, vector_drawings=0, captions=[])) == "no_signal"
    )


def test_classify_ignores_noisy_vector_drawings_without_caption_or_raster():
    """Regressions-Guard: Layout-Vektoren allein erzeugen keine Figur-Klasse."""
    signals = PageVisualSignals(page=1, raster_images=0, vector_drawings=56, captions=[])

    assert classify_page_signals(signals) == "no_signal"


def test_classify_vector_count_never_upgrades_the_class():
    """Hoher vector_drawings-Count darf keine Klasse hochstufen (alter Dominanz-Fehler)."""
    caption = find_figure_captions("Abb. 2: Schema.", page=7)
    # Caption ohne Raster bleibt captioned_no_raster, egal wie viele Vektorpfade.
    assert (
        classify_page_signals(PageVisualSignals(page=7, raster_images=0, vector_drawings=99, captions=caption))
        == "captioned_no_raster"
    )
    # Raster ohne Caption bleibt raster_uncaptioned, egal wie viele Vektorpfade.
    assert (
        classify_page_signals(PageVisualSignals(page=8, raster_images=1, vector_drawings=99, captions=[]))
        == "raster_uncaptioned"
    )
    # Caption + Raster bleibt captioned_raster, egal wie viele Vektorpfade.
    assert (
        classify_page_signals(PageVisualSignals(page=9, raster_images=1, vector_drawings=99, captions=caption))
        == "captioned_raster"
    )


def test_warning_only_fires_for_no_signal_and_is_honest_about_vector():
    """no_signal = weder Caption noch Raster; Vektorpfade zaehlen nicht als Signal.

    Der Warning-Text darf nicht "no visual signal" behaupten, solange
    vector_drawings (Layout-Rohsignal) im selben Report weiter ausgegeben wird.
    """
    assert warning_for_classification("no_signal") == "no caption or raster signal"
    assert warning_for_classification("captioned_raster") is None
    assert warning_for_classification("captioned_no_raster") is None
    assert warning_for_classification("raster_uncaptioned") is None


def test_report_has_no_hardcoded_literature_paths():
    assert report_has_no_hardcoded_literature_paths()


def test_chunk_summary_rows_exclude_chunk_text_from_report_payload():
    chunks = [Chunk("Abschnitt 1", "sensitive full chunk text", 0, page_start=1, page_end=3)]

    rows = chunk_summary_rows(chunks)

    assert rows == [{"title": "Abschnitt 1", "index": 0, "page_start": 1, "page_end": 3}]
    assert "sensitive full chunk text" not in str(rows)


def _build_pdf_with_textless_middle_page(path: Path) -> None:
    """PDF: physische Seite 0 (Text), Seite 1 (leer → von pdftotext verworfen),
    Seite 2 (Text + Raster-Bild). Kein /PageLabels → Fallback-Pfad in #78."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    p0 = doc.new_page()
    p0.insert_text((72, 72), "Seite null mit ausreichend Text fuer pdftotext hier.")
    doc.new_page()  # physische Seite 1: absichtlich leer/textlos
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Seite zwei mit Text und einem eingebetteten Bild.")
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 24, 24))
    pix.set_rect(fitz.IRect(0, 0, 24, 24), (200, 40, 40))
    p2.insert_image(fitz.Rect(200, 200, 260, 260), stream=pix.tobytes("png"))
    doc.save(str(path))
    doc.close()


def test_page_visual_signals_maps_textpage_past_textless_page(tmp_path):
    """#78: pdf_to_pages verwirft textlose Seiten und nummeriert lückenlos neu.

    Das Raster-Bild liegt auf physischer Seite 2 (= text-Seite 2, weil die leere
    physische Seite 1 verworfen wird). Der Fallback `page_number - 1` zeigte auf
    die leere Seite → 0 Raster. Korrektes Mapping (has_text-Zählung) findet das Bild.
    """
    from generative.pipeline import pdf_chunker

    pdf_path = tmp_path / "textless_middle.pdf"
    _build_pdf_with_textless_middle_page(pdf_path)

    pages = pdf_chunker.pdf_to_pages(pdf_path)
    page_numbers = [n for n, _ in pages]
    assert page_numbers == [1, 2], "Erwartet: leere Seite verworfen, lückenlos [1, 2]"

    signals = _page_visual_signals(pdf_path, pages)
    by_page = {s.page: s for s in signals}

    assert by_page[2].raster_images >= 1, "Raster-Bild muss text-Seite 2 zugeordnet werden, nicht der leeren Seite"
