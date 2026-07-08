# tests/extractive/test_orchestrator.py
"""Verdrahtungs-Tests fuer die Orchestrator-Note-Erzeugung (#167 b/c)."""

from unittest.mock import patch

from extractive import orchestrator


def test_build_notes_passes_detected_language():
    """Fix #167b: die erkannte Sprache landet als sumy-Sprachname im LexRank-Aufruf
    (frueher immer 'english' — deutsche PDFs bekamen den englischen Tokenizer)."""
    concepts = [{"name": "Informationskompetenz", "type": "Concept", "page": 1}]
    with patch("extractive.orchestrator.extract_body_for_concept") as mock_ex:
        mock_ex.return_value = ["Ein Satz ueber Informationskompetenz im Text."]
        orchestrator._build_notes(concepts, "ft", ["ft"], "de", "src.pdf", "2026-07-08")
    _, kwargs = mock_ex.call_args
    assert kwargs.get("language") == "german"


def test_build_notes_maps_sentence_to_physical_page():
    """Fix #167c: ein real auf Seite 2 stehender Satz wird mit (S. 2) verankert,
    nicht mit der Chunk-Startseite (hier Fallback 1)."""
    page_texts = [
        "Alpha concept is introduced on the very first page of the document here.",
        "Beta discussion continues about the alpha concept on the second page exclusively.",
    ]
    fulltext = "\n".join(page_texts)
    with patch("extractive.orchestrator.extract_body_for_concept") as mock_ex:
        mock_ex.return_value = ["Beta discussion continues about the alpha concept on the second page exclusively."]
        notes = orchestrator._build_notes(
            [{"name": "alpha", "type": "Concept", "page": 1}],
            fulltext,
            page_texts,
            "en",
            "src.pdf",
            "2026-07-08",
        )
    assert notes, "keine Note erzeugt"
    assert "(S. 2)" in notes[0].extracted_body[0], notes[0].extracted_body
    assert notes[0].source_anchors[0]["page"] == 2
