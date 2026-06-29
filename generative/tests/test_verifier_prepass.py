"""Tests für deterministischen Pre-Pass in verifier.run().

Quality-Garantie: Pre-Pass mit Fuzzy ≥98 + Mindestlänge 30 löst hoch-konfidente
Anker ohne LLM-Call auf. LLM-Pfad bleibt Safety-Net für Edge-Cases.
"""
from __future__ import annotations
from unittest.mock import patch


from generative.agents import verifier
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor


def _draft(anchors: list[TextAnchor]) -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title="Test",
        body="",
        source_anchors=anchors,
        related=[],
        tags=[],
        synthesis_confidence="medium",
    )


CHUNK_WITH_MARKERS = (
    "[S. 1]\nIrgendein Einleitungstext der nicht relevant ist.\n"
    "[S. 2]\nHier steht ein langer charakteristischer Satz mit prägnantem Inhalt zur Verifikation.\n"
    "[S. 3]\nWeiterer Absatz mit anderen Aussagen die nicht zum Quote passen.\n"
)


def test_prepass_full_resolve_skips_llm():
    anchors = [
        TextAnchor(
            quote="Hier steht ein langer charakteristischer Satz mit prägnantem Inhalt zur Verifikation.",
            page=None,
        ),
    ]
    draft = _draft(anchors)

    with patch.object(verifier, "call_claude") as mock_llm, \
         patch.object(verifier, "_semantic_find_page", return_value=None):
        result = verifier.run(draft, CHUNK_WITH_MARKERS)

    assert not mock_llm.called, "LLM darf nicht gerufen werden wenn alle Anker pre-resolved sind"
    assert len(result.source_anchors) == 1
    assert result.source_anchors[0].page == "S. 2"
    assert result.source_anchors[0].fuzzy_page is None


def test_prepass_partial_only_unresolved_to_llm():
    anchors = [
        TextAnchor(
            quote="Hier steht ein langer charakteristischer Satz mit prägnantem Inhalt zur Verifikation.",
            page=None,
        ),
        TextAnchor(
            quote="Eine ganz andere nicht im Text vorkommende Aussage die der Verifier prüfen muss.",
            page=None,
        ),
    ]
    draft = _draft(anchors)

    llm_raw = (
        "all_verified: false\n"
        "<!--ANCHOR-->\n"
        "page:\n"
        "verified: false\n"
        "<!--QUOTE-->\n"
        "Eine ganz andere nicht im Text vorkommende Aussage die der Verifier prüfen muss.\n"
        "<!--END-->\n"
    )

    with patch.object(verifier, "call_claude", return_value=llm_raw) as mock_llm, \
         patch.object(verifier, "_semantic_find_page", return_value=None):
        result = verifier.run(draft, CHUNK_WITH_MARKERS)

    assert mock_llm.call_count == 1
    prompt = mock_llm.call_args[0][0]
    # Anker-Liste-Block isolieren (vor "## Originaltext"), damit chunk_text-Inhalte
    # nicht False-Positive triggern
    anchors_block = prompt.split("## Originaltext")[0]
    assert "charakteristischer Satz" not in anchors_block, "Pre-resolved Anker darf nicht im Anchor-Listing sein"
    assert "ganz andere nicht im Text" in anchors_block

    kept_quotes = {a.quote for a in result.source_anchors}
    assert any("charakteristischer Satz" in q for q in kept_quotes), "Pre-resolved Anker muss erhalten bleiben"


def test_short_quote_below_min_len_goes_to_llm():
    anchors = [TextAnchor(quote="kurz aber im text", page=None)]
    chunk = "[S. 1]\nkurz aber im text steht hier\n"
    draft = _draft(anchors)

    with patch.object(verifier, "call_claude", return_value="all_verified: false\n<!--END-->\n") as mock_llm, \
         patch.object(verifier, "_semantic_find_page", return_value=None):
        verifier.run(draft, chunk)

    assert mock_llm.call_count == 1, "Quote <30 chars → kein Pre-Pass, LLM-Pfad erwartet"


def test_low_fuzzy_score_not_prepassed():
    anchors = [
        TextAnchor(
            quote="Vollkommen abweichende Formulierung die nirgendwo im Originaltext vorkommt.",
            page=None,
        ),
    ]
    draft = _draft(anchors)

    with patch.object(verifier, "call_claude", return_value="all_verified: false\n<!--END-->\n") as mock_llm, \
         patch.object(verifier, "_semantic_find_page", return_value=None):
        verifier.run(draft, CHUNK_WITH_MARKERS)

    assert mock_llm.call_count == 1, "Score <98 → LLM-Pfad erwartet"


def test_exact_substring_sets_page_fuzzy_sets_fuzzy_page():
    # Exact-Match-Pfad: quote ist 1:1 im chunk → page
    exact_quote = "Hier steht ein langer charakteristischer Satz mit prägnantem Inhalt zur Verifikation."
    # Fuzzy-Pfad: Quote leicht abgewandelt (z.B. fehlender Punkt am Ende) — Fuzzy ≥98, kein Exact
    fuzzy_quote = "Hier steht ein langer charakteristischer Satz mit prägnantem Inhalte zur Verifikation"
    anchors = [
        TextAnchor(quote=exact_quote, page=None),
        TextAnchor(quote=fuzzy_quote, page=None),
    ]
    draft = _draft(anchors)

    with patch.object(verifier, "call_claude"), \
         patch.object(verifier, "_semantic_find_page", return_value=None):
        result = verifier.run(draft, CHUNK_WITH_MARKERS)

    by_quote = {a.quote: a for a in result.source_anchors}
    assert by_quote[exact_quote].page == "S. 2"
    assert by_quote[exact_quote].fuzzy_page is None, "Exact-Match → page, nicht fuzzy_page"
    # Fuzzy-Match: page=None, fuzzy_page gesetzt — Critic sieht es nicht als verifiziert
    assert by_quote[fuzzy_quote].page is None, "Fuzzy-Match darf page nicht setzen"
    assert by_quote[fuzzy_quote].fuzzy_page == "S. 2"


def test_llm_fail_preserves_pre_resolved_and_original_unresolved():
    exact_quote = "Hier steht ein langer charakteristischer Satz mit prägnantem Inhalt zur Verifikation."
    other_quote = "Eine andere Aussage mit Original-Page vom Extractor die nicht im Text steht."
    anchors = [
        TextAnchor(quote=exact_quote, page=None),
        TextAnchor(quote=other_quote, page="S. 9", fuzzy_page=None),  # Extractor hatte schon Page
    ]
    draft = _draft(anchors)

    def boom(*a, **kw):
        raise RuntimeError("LLM down")

    with patch.object(verifier, "call_claude", side_effect=boom), \
         patch.object(verifier, "_semantic_find_page", return_value=None):
        result = verifier.run(draft, CHUNK_WITH_MARKERS)

    by_quote = {a.quote: a for a in result.source_anchors}
    assert by_quote[exact_quote].page == "S. 2", "Pre-resolved überlebt LLM-Fail"
    assert by_quote[other_quote].page == "S. 9", "Original-Extractor-Page bleibt erhalten"
    assert any("Verifier nicht ausgeführt" in f for f in result.quality_flags)


def test_prepass_uses_last_marker_before_match():
    anchors = [
        TextAnchor(
            quote="Hier steht ein langer charakteristischer Satz mit prägnantem Inhalt zur Verifikation.",
            page=None,
        ),
    ]
    draft = _draft(anchors)

    with patch.object(verifier, "call_claude"), \
         patch.object(verifier, "_semantic_find_page", return_value=None):
        result = verifier.run(draft, CHUNK_WITH_MARKERS)

    assert result.source_anchors[0].page == "S. 2", "Letzter Marker vor Match-Position"


# --- Tier-3: Semantischer Pre-Pass ---

def test_semantic_prepass_resolves_without_llm():
    """Wenn semantic_find_page eine Seite liefert, darf LLM nicht gerufen werden."""
    anchors = [
        TextAnchor(
            quote="Eine Paraphrase die semantisch passt aber nicht fuzzy-matched.",
            page=None,
        ),
    ]
    draft = _draft(anchors)

    with patch.object(verifier, "call_claude") as mock_llm, \
         patch.object(verifier, "_semantic_find_page", return_value="S. 3"):
        result = verifier.run(draft, CHUNK_WITH_MARKERS)

    assert not mock_llm.called, "LLM darf nicht gerufen werden wenn semantic aufgelöst"
    assert len(result.source_anchors) == 1
    assert result.source_anchors[0].fuzzy_page == "S. 3"
    assert result.source_anchors[0].page is None, "Semantic liefert fuzzy_page, nicht page"


def test_semantic_prepass_only_for_unresolved():
    """Exact-pre-resolved Anker gehen nicht durch semantic (wären Doppelaufruf)."""
    exact_quote = "Hier steht ein langer charakteristischer Satz mit prägnantem Inhalt zur Verifikation."
    para_quote = "Paraphrase ohne Fuzzy-Match aber semantisch verwandt zum Originaltext."
    anchors = [
        TextAnchor(quote=exact_quote, page=None),
        TextAnchor(quote=para_quote, page=None),
    ]
    draft = _draft(anchors)

    semantic_calls: list[str] = []

    def capture_semantic(quote, chunk_text, **kw):
        semantic_calls.append(quote)
        return "S. 3" if para_quote in quote else None

    with patch.object(verifier, "call_claude") as mock_llm, \
         patch.object(verifier, "_semantic_find_page", side_effect=capture_semantic):
        result = verifier.run(draft, CHUNK_WITH_MARKERS)

    assert not mock_llm.called
    # Exact-Quote wurde pre-resolved (Fuzzy-Pass) → semantic darf nicht darauf gerufen werden
    assert exact_quote not in semantic_calls, "Exact-resolved Anker soll semantic überspringen"
    assert para_quote in semantic_calls, "Unresolved Anker muss durch semantic"
    by_quote = {a.quote: a for a in result.source_anchors}
    assert by_quote[exact_quote].page == "S. 2"
    assert by_quote[para_quote].fuzzy_page == "S. 3"


def test_semantic_no_match_falls_through_to_llm():
    """Wenn semantic None liefert, geht der Anker wie bisher an LLM."""
    anchors = [
        TextAnchor(
            quote="Eine Paraphrase ohne semantischen Match die an LLM geht.",
            page=None,
        ),
    ]
    draft = _draft(anchors)
    llm_raw = "all_verified: false\n<!--END-->\n"

    with patch.object(verifier, "call_claude", return_value=llm_raw) as mock_llm, \
         patch.object(verifier, "_semantic_find_page", return_value=None):
        verifier.run(draft, CHUNK_WITH_MARKERS)

    assert mock_llm.call_count == 1, "Kein semantic-Match → LLM-Pfad"
