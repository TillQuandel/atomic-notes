"""Tests für Page-Index + Claim-Quellfenster (Faithfulness-Gate E1, #69).

Pure Helfer ohne Pipeline-Verdrahtung: `build_page_index` splittet PDF-Volltext
(mit `[S. N]`-Markern aus `pages_to_marked_text`) in ein Seiten-Dict,
`claim_source_window` liefert daraus das Quellfenster für einen auf eine
Seite verankerten Claim (Seite + Nachbarseiten in der sortierten Key-Liste,
NICHT arithmetisch page±1 — Kapitel-Auszüge haben Seitenlücken).
"""

from __future__ import annotations

from generative.pipeline.pdf_chunker import pages_to_marked_text
from generative.pipeline.page_index import build_page_index, claim_source_window


# ---- build_page_index ------------------------------------------------------


def test_roundtrip_with_pages_to_marked_text():
    text = pages_to_marked_text([(1, "Erster Seiteninhalt."), (2, "Zweiter Seiteninhalt.")])
    idx = build_page_index(text)
    assert idx[1] == "Erster Seiteninhalt."
    assert idx[2] == "Zweiter Seiteninhalt."


def test_inline_reference_does_not_create_page():
    # „vgl. [S. 12]" steht mitten in der Zeile — KEIN eigener Marker, bleibt
    # als Text erhalten (dokumentierte Fehlerklasse: Inline-Verweis != Seitengrenze).
    text = pages_to_marked_text([(5, "Ein Satz mit Verweis vgl. [S. 12] auf andere Stelle.")])
    idx = build_page_index(text)
    assert list(idx.keys()) == [5]
    assert "vgl. [S. 12]" in idx[5]


def test_empty_string_returns_empty_dict():
    assert build_page_index("") == {}


def test_text_without_markers_returns_empty_dict():
    assert build_page_index("Nur Fließtext ohne jeden Seitenmarker.") == {}


def test_duplicate_page_number_concatenated():
    text = "\n\n[S. 1]\n\nErster Teil." + "\n\n[S. 1]\n\nZweiter Teil."
    idx = build_page_index(text)
    assert idx[1] == "Erster Teil.\nZweiter Teil."


def test_text_before_first_marker_is_discarded():
    text = "Verworfener Text vor dem ersten Marker." + "\n\n[S. 1]\n\nEchter Seiteninhalt."
    idx = build_page_index(text)
    assert list(idx.keys()) == [1]
    assert "Verworfener" not in idx[1]


# ---- claim_source_window ---------------------------------------------------


def _idx_with_gap() -> dict[int, str]:
    text = pages_to_marked_text(
        [
            (159, "Inhalt Seite 159."),
            (161, "Inhalt Seite 161."),
            (200, "Inhalt Seite 200."),
        ]
    )
    return build_page_index(text)


def test_gap_neighbors_are_index_neighbors_not_arithmetic():
    idx = _idx_with_gap()
    window = claim_source_window(idx, 161)
    assert "[S. 159]" in window
    assert "[S. 161]" in window
    assert "[S. 200]" in window


def test_first_page_only_has_one_sided_neighbor():
    idx = _idx_with_gap()
    window = claim_source_window(idx, 159)
    assert window is not None
    assert "[S. 159]" in window
    assert "[S. 161]" in window
    assert "[S. 200]" not in window


def test_last_page_only_has_one_sided_neighbor():
    idx = _idx_with_gap()
    window = claim_source_window(idx, 200)
    assert window is not None
    assert "[S. 200]" in window
    assert "[S. 161]" in window
    assert "[S. 159]" not in window


def test_unknown_page_returns_none():
    idx = _idx_with_gap()
    assert claim_source_window(idx, 999) is None


def test_neighbors_zero_returns_only_own_page():
    idx = _idx_with_gap()
    window = claim_source_window(idx, 161, neighbors=0)
    assert window == "[S. 161]\nInhalt Seite 161."


def test_window_format_markers_ascending():
    idx = _idx_with_gap()
    window = claim_source_window(idx, 161)
    pos_159 = window.index("[S. 159]")
    pos_161 = window.index("[S. 161]")
    pos_200 = window.index("[S. 200]")
    assert pos_159 < pos_161 < pos_200
    assert window == ("[S. 159]\nInhalt Seite 159.\n\n[S. 161]\nInhalt Seite 161.\n\n[S. 200]\nInhalt Seite 200.")


def test_duplicate_page_number_in_window():
    text = "\n\n[S. 1]\n\nA-Teil." + "\n\n[S. 1]\n\nB-Teil." + "\n\n[S. 2]\n\nZweite Seite."
    idx = build_page_index(text)
    window = claim_source_window(idx, 1, neighbors=1)
    assert "A-Teil.\nB-Teil." in window
    assert "[S. 2]" in window
