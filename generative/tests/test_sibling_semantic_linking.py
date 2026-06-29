"""Tests für semantisches Sibling-Candidate-Ranking (Stage B).

Wurzel: Stage B gated Pipeline-Geschwister bisher über rohen Titel-/Alias-Token-
Overlap (`overlap >= 1`). Zwei semantisch fast identische Notes EINES Laufs mit
lexikalisch disjunkten Titeln ("Wissensorganisation" ↔ "Semantisches Retrieval mit
Assoziationsnetz", Body-cos 0,97) teilten 0 Tokens → Geschwister verworfen → beide
`related: []`. Fix: zusätzlich Body-Embedding-Cosine als Kandidaten-Signal (additiv —
Token-Treffer bleiben unverändert, kein Regressionsrisiko). Schwelle 0,85 empirisch
kalibriert (verwandt 0,97–0,99, fremd 0,73–0,76).
"""

from __future__ import annotations

from generative.agents.cross_reference import _rank_sibling_candidates, _tokens


def _d(title, body="b", aliases=None):
    from generative.schemas.atomic_note import AtomicNoteDraft

    return AtomicNoteDraft(
        title=title,
        body=body,
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="low",
        aliases=aliases or [],
    )


def _q(draft):
    q = _tokens(draft.title)
    for a in draft.aliases:
        q |= _tokens(a)
    return q


def test_token_overlap_sibling_included_without_embedding():
    # Geschwister mit gemeinsamem Token ("ISP") → Kandidat; cosine_fn NICHT aufgerufen.
    draft = _d("ISP Stage Collection")
    sibs = {"ISP Stage Exploration": _d("ISP Stage Exploration")}
    called = []

    def cos_fn(s):
        called.append(s)
        return 0.0

    out = _rank_sibling_candidates(draft, sibs, _q(draft), cos_fn, threshold=0.85)
    assert [t for t, _ in out] == ["ISP Stage Exploration"]
    assert called == []  # Token-Treffer → kein Embedding nötig


def test_lexically_disjoint_but_semantic_included():
    # Der echte Fall: 0 gemeinsame Tokens, aber Body-cos 0,97 → Kandidat.
    draft = _d("Wissensorganisation")
    sibs = {"Semantisches Retrieval mit Assoziationsnetz": _d("Semantisches Retrieval mit Assoziationsnetz")}
    out = _rank_sibling_candidates(draft, sibs, _q(draft), lambda s: 0.97, threshold=0.85)
    assert [t for t, _ in out] == ["Semantisches Retrieval mit Assoziationsnetz"]


def test_lexically_disjoint_below_threshold_excluded():
    # 0 Tokens UND Body-cos unter Schwelle (fremdes Thema, 0,76) → kein Kandidat.
    draft = _d("Wissensorganisation")
    sibs = {"ADKAR Ability": _d("ADKAR Ability")}
    out = _rank_sibling_candidates(draft, sibs, _q(draft), lambda s: 0.76, threshold=0.85)
    assert out == []


def test_self_excluded():
    draft = _d("Wissensorganisation")
    sibs = {"Wissensorganisation": draft}
    out = _rank_sibling_candidates(draft, sibs, _q(draft), lambda s: 0.99, threshold=0.85)
    assert out == []


def test_lexical_ranks_above_semantic():
    # Token-Treffer (starkes Signal) muss vor reinem Embedding-Treffer ranken.
    draft = _d("ISP Stage Collection")
    sibs = {
        "ISP Stage Exploration": _d("ISP Stage Exploration"),  # Token-Overlap ("ISP","Stage")
        "Affektives Paradigma der Suche": _d("Affektives Paradigma der Suche"),  # nur semantisch
    }
    out = _rank_sibling_candidates(draft, sibs, _q(draft), lambda s: 0.95, threshold=0.85)
    assert out[0][0] == "ISP Stage Exploration"
    assert "Affektives Paradigma der Suche" in [t for t, _ in out]


def test_empty_siblings_returns_empty():
    draft = _d("X")
    assert _rank_sibling_candidates(draft, None, _q(draft), lambda s: 0.99) == []
    assert _rank_sibling_candidates(draft, {}, _q(draft), lambda s: 0.99) == []


def test_caps_at_five():
    draft = _d("Topic")
    sibs = {f"Sib {i}": _d(f"Sib {i}") for i in range(8)}  # alle semantisch
    out = _rank_sibling_candidates(draft, sibs, _q(draft), lambda s: 0.9, threshold=0.85)
    assert len(out) == 5
