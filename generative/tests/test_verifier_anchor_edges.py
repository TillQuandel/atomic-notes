"""Tests für Anker-/Seiten-Edge-Cases im Verifier (Issue #77).

Drei Edge-Cases:
  A — Text VOR dem ersten `[S. N]`-Marker (Chunk startet mitten auf einer Seite)
      darf in `_build_page_sections` nicht verworfen werden.
  B — NBSP in Body-Quotes (` `) darf in `sync_anchors_from_body` nicht zu
      Duplikat-Ankern führen (Quote-Vergleich NFKC-normalisiert).
  C — Degeneriertes `quote_clean` (nur Whitespace/Sonderzeichen + 1 Realzeichen)
      darf in `_fuzzy_find_page` keine False-Positive-Seite matchen.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from generative.agents import verifier
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor


def _draft(body: str, anchors: list[TextAnchor]) -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title="Test",
        body=body,
        source_anchors=anchors,
        related=[],
        tags=[],
        synthesis_confidence="medium",
    )


# --- A: Text vor erstem Marker ---


def test_build_page_sections_includes_leading_text_before_first_marker():
    """Chunk startet mitten auf einer Seite; erster Marker ist [S. 6].

    Marker stehen an Seitenanfängen (pdf_chunker.pages_to_marked_text), also
    gehört der führende Text zu Seite 5. Er darf nicht verworfen werden — sonst
    bekommen Zitate in diesem Bereich keine Embeddings und semantic liefert None.
    """
    chunk = "Fuehrender Satz vor jedem Marker steht hier.\n\n[S. 6]\n\nInhalt von Seite sechs.\n"

    class _FakeModel:
        def encode(self, sents, **kw):
            return np.ones((len(sents), 3), dtype="float32")

    def _fake_sentences(text: str) -> list[str]:
        return [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

    with (
        patch("generative.embeddings._MODEL", object()),
        patch("generative.embeddings._model", return_value=_FakeModel()),
        patch("generative.embeddings._sentences", _fake_sentences),
    ):
        sections = verifier._build_page_sections(chunk)

    assert sections is not None
    pages = [p for p, _ in sections]
    assert "5" in pages, "Führende Section vor erstem Marker muss mit Fallback-Seite (first-1) erscheinen"
    assert "6" in pages


def test_build_page_sections_no_zero_page_when_first_marker_is_one():
    """Erster Marker [S. 1]: führender Text hätte Seite 0 — nie emittieren."""
    chunk = "Vorlauftext ohne echte Seitenzahl.\n\n[S. 1]\n\nErster Seiteninhalt.\n"

    class _FakeModel:
        def encode(self, sents, **kw):
            return np.ones((len(sents), 3), dtype="float32")

    def _fake_sentences(text: str) -> list[str]:
        return [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

    with (
        patch("generative.embeddings._MODEL", object()),
        patch("generative.embeddings._model", return_value=_FakeModel()),
        patch("generative.embeddings._sentences", _fake_sentences),
    ):
        sections = verifier._build_page_sections(chunk)

    pages = [p for p, _ in sections] if sections else []
    assert "0" not in pages, "Seite 0 ist nie zitierfähig"


# --- B: NBSP-Mismatch ---


def test_sync_anchors_nbsp_quote_not_duplicated():
    """Body-Quote mit NBSP == existierender Anker mit normalem Space → kein Duplikat."""
    quote_regular = "foo bar baz qux zap wibble"
    body = 'Text „foo bar baz qux zap wibble" (S. 5). Mehr Text.'
    draft = _draft(body, [TextAnchor(quote=quote_regular, page=None, fuzzy_page=None)])

    verifier.sync_anchors_from_body(draft)

    assert len(draft.source_anchors) == 1, "NBSP-Variante darf keinen Duplikat-Anker erzeugen"
    assert draft.source_anchors[0].fuzzy_page == "S. 5", "Page muss beim bestehenden Anker nachgetragen werden"


# --- C: Degeneriertes quote_clean ---


def test_fuzzy_find_page_ignores_degenerate_short_clean_quote():
    """Roh-Quote ≥15 Zeichen, aber nach Strip nur 1 Realzeichen → keine Seite.

    Sonst matcht partial_ratio das 1-Zeichen-quote_clean mit Score 100 an
    beliebiger Stelle → False-Positive-Seite.
    """
    quote = "„" + " " * 20 + 'x"'  # raw len 23, quote_clean nach Strip = "x"
    assert len(quote) >= 15
    text = "[S. 4]\n" + " " * 20 + "x irgendwo im Fließtext.\n"

    assert verifier._fuzzy_find_page(quote, text) is None


def test_fuzzy_find_page_still_matches_normal_quote():
    """Regressions-Guard: normale Quotes matchen weiterhin."""
    text = "[S. 7]\nEin langer charakteristischer Satz mit prägnantem Inhalt steht hier.\n"
    quote = "Ein langer charakteristischer Satz mit prägnantem Inhalt"

    assert verifier._fuzzy_find_page(quote, text) == "S. 7"
