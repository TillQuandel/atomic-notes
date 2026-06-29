"""Tests für #8: Detektions-Flag bei hoher Body-Überlappung zwischen DISTINKTEN Notes.

Zwei empirische Gates (Session 2026-06-23, Ebner-Audit) zeigten: Geschwister-Notes mit
hoher Body-Cosine sind weder mergebar (distinkte Konzepte: Kirkpatrick-Modell = Theorie
vs. Satisfaction-Learning-Dissoziation = Befund) noch satz-strippbar (Redundanz
paraphrasiert, nicht dupliziert — exakt 0/10, fuzzy≥0.93 nur 1/10 Sätze). Statt eines
riskanten Strips: ein seiteneffekt-freier Flag auf BEIDE Notes, der den menschlichen
Reviewer auf die Überlappung hinweist. Kein Body-Eingriff, kein Kollabieren.

flag_redundant_siblings() läuft NACH resolve_sibling_dups + dedup_hub_subconcepts (echte
Dups/Hub-Sub schon behandelt) und vor dem Writer (Flag landet im Frontmatter).
"""

import pytest

from generative.orchestrator import flag_redundant_siblings
from generative.schemas.atomic_note import AtomicNoteDraft


def _draft(title, *, body="", action="create", extend_path=None):
    return AtomicNoteDraft(
        title=title,
        body=body or f"Body von {title}",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high",
        action=action,
        extend_path=extend_path,
    )


def _cos_map(pairs):
    """Injizierbare body_cosine_fn aus einem {(i,j): cos}-Dict (symmetrisch)."""

    def fn(i, j):
        return pairs.get((i, j), pairs.get((j, i), 0.0))

    return fn


def test_two_distinct_create_drafts_above_threshold_flag_both():
    d_a = _draft("Kirkpatrick-Modell")
    d_b = _draft("Satisfaction-Learning-Dissoziation")

    kept, n = flag_redundant_siblings([d_a, d_b], threshold=0.90, body_cosine_fn=_cos_map({(0, 1): 0.967}))

    assert n == 1
    assert len(kept) == 2  # nichts kollabiert — beide bleiben distinkt
    # BEIDE bekommen den Flag, jeweils auf den anderen verweisend
    assert any("[[Satisfaction-Learning-Dissoziation]]" in f for f in d_a.quality_flags)
    assert any("[[Kirkpatrick-Modell]]" in f for f in d_b.quality_flags)
    # Flag enthält Review-Hinweis (kein Strip)
    assert any("Review" in f for f in d_a.quality_flags)


def test_below_threshold_no_flag():
    d_a = _draft("Webinar als Lernformat")
    d_b = _draft("Kirkpatrick-Modell")

    kept, n = flag_redundant_siblings([d_a, d_b], threshold=0.90, body_cosine_fn=_cos_map({(0, 1): 0.70}))

    assert n == 0
    assert d_a.quality_flags == []
    assert d_b.quality_flags == []


def test_extend_drafts_excluded():
    # extend-Drafts werden bereits von resolve_sibling_dups / write_note behandelt —
    # hier nicht doppelt flaggen, selbst wenn die Body-Cosine hoch wäre.
    d_a = _draft("Konzept A")
    d_b = _draft("Konzept B")
    d_c = _draft("Vault-Dup", action="extend", extend_path="Bestehende Vault-Note")

    # alle Paare hoch — aber d_c ist extend und darf nicht verglichen werden
    kept, n = flag_redundant_siblings(
        [d_a, d_b, d_c], threshold=0.90, body_cosine_fn=_cos_map({(0, 1): 0.95, (0, 2): 0.95, (1, 2): 0.95})
    )

    assert n == 1  # nur das create/create-Paar (a,b)
    assert d_c.quality_flags == []  # extend nie geflaggt
    assert any("[[Konzept B]]" in f for f in d_a.quality_flags)


def test_single_create_draft_noop():
    d_a = _draft("Allein")
    kept, n = flag_redundant_siblings([d_a], threshold=0.90, body_cosine_fn=_cos_map({}))
    assert n == 0
    assert d_a.quality_flags == []


def test_idempotent_no_duplicate_flags():
    d_a = _draft("A")
    d_b = _draft("B")
    cos = _cos_map({(0, 1): 0.95})

    flag_redundant_siblings([d_a, d_b], threshold=0.90, body_cosine_fn=cos)
    flag_redundant_siblings([d_a, d_b], threshold=0.90, body_cosine_fn=cos)

    # zweimal laufen darf den Flag nicht duplizieren
    a_redund = [f for f in d_a.quality_flags if "Überlappung mit [[B]]" in f]
    assert len(a_redund) == 1


def test_one_high_one_low_among_three():
    d_a = _draft("A")
    d_b = _draft("B")
    d_c = _draft("C")
    # a~b hoch, a~c und b~c niedrig
    cos = _cos_map({(0, 1): 0.95, (0, 2): 0.40, (1, 2): 0.42})

    kept, n = flag_redundant_siblings([d_a, d_b, d_c], threshold=0.90, body_cosine_fn=cos)
    assert n == 1
    assert d_c.quality_flags == []  # C überlappt mit niemandem
    assert any("[[B]]" in f for f in d_a.quality_flags)
    assert any("[[A]]" in f for f in d_b.quality_flags)


@pytest.mark.slow
def test_real_embeddings_default_path():
    """Ohne Injection: echtes Embedding-Modell, zwei fast identische Bodies → Flag.
    Validiert dass der Default-Pfad (config-Schwelle, embeddings.embed_body/cosine) wirkt."""
    shared = (
        "Lernumgebungen lassen sich nach Synchronizität und Modalität "
        "klassifizieren. Die Meta-Analyse fand einen kleinen positiven Effekt "
        "auf die Lernleistung mit großer Heterogenität zwischen den Studien."
    )
    d_a = _draft("Note A", body=shared)
    d_b = _draft("Note B", body=shared + " Geringfügige Ergänzung.")

    kept, n = flag_redundant_siblings([d_a, d_b])  # Default-Schwelle, echtes Modell
    assert n == 1
    assert any("Überlappung" in f for f in d_a.quality_flags)
