"""Tests für den cross-lingualen Rettungsanker in planner.filter_hallucinated.

Der lexikalische Token-Coverage-Filter ist sprachblind: ein deutscher (paraphrasierter)
Konzept-Titel hat null wörtlichen Overlap mit einer englischen Quelle und wurde fälschlich
als „halluziniert" verworfen (Ebner-Run 2026-06-23: der Paper-Kernbefund „Lern-Zufriedenheits-
Dissoziation" starb so). Der semantische Präsenz-Check (MAX-Cosine) rettet solche Konzepte,
bevor sie verworfen werden — als reiner OR-Kanal (kann nur retten, nie zusätzlich verwerfen).
"""

from generative.agents.planner import filter_hallucinated
from generative.schemas.atomic_note import ConceptPlan, ConceptItem


def _plan(*titles):
    return ConceptPlan(
        source_title="S",
        source_summary="zwei Sätze worum es geht",
        concepts=[
            ConceptItem(
                title=t,
                priority="high",
                chapter="",
                action="create",
                extend_path=None,
                category="conceptual",
                origin="primary",
                cited_authors=[],
            )
            for t in titles
        ],
    )


def test_lexically_present_kept_without_consulting_semantic():
    # Titel lexikalisch im Text → semantischer Kanal darf gar nicht erst aufgerufen werden.
    calls = []

    def _spy(title, text):
        calls.append(title)
        return 0.0  # würde verwerfen, falls fälschlich konsultiert

    kept, rejected = filter_hallucinated(
        _plan("Webinar"), "A meta-analysis about Webinar effectiveness in learning.", semantic_presence_fn=_spy
    )
    assert [c.title for c in kept.concepts] == ["Webinar"]
    assert rejected == []
    assert calls == []  # lexikalischer Pass → kein Embedding-Call


def test_crosslingual_concept_rescued_by_high_semantic_presence():
    # DE-Titel, EN-Quelle → lexikalische Coverage 0, aber semantisch hoch → GERETTET.
    kept, rejected = filter_hallucinated(
        _plan("Lern-Zufriedenheits-Dissoziation"),
        "Learning and satisfaction were negatively associated in all three conditions.",
        semantic_presence_fn=lambda t, txt: 0.83,
    )
    assert [c.title for c in kept.concepts] == ["Lern-Zufriedenheits-Dissoziation"]
    assert rejected == []


def test_genuine_hallucination_rejected_by_low_semantic_presence():
    # Weder lexikalisch noch semantisch präsent → verworfen (Default-Schwelle 0.50).
    kept, rejected = filter_hallucinated(
        _plan("Quantenverschränkung in Photonenpaaren"),
        "A meta-analysis about webinars and student satisfaction.",
        semantic_presence_fn=lambda t, txt: 0.36,
    )
    assert kept.concepts == []
    assert "Quantenverschränkung in Photonenpaaren" in rejected


def test_blacklist_rejected_even_when_semantic_rescues_lexical():
    # Generischer Titel: vom semantischen Kanal lexikalisch gerettet, dann aber von der
    # Generika-Blacklist verworfen — Reihenfolge muss erhalten bleiben.
    kept, rejected = filter_hallucinated(
        _plan("System"), "irrelevanter Text ohne das Wort", semantic_presence_fn=lambda t, txt: 0.99
    )
    assert kept.concepts == []
    assert "System" in rejected


def test_rescue_respects_configurable_threshold():
    # Schwelle ist die config-Konstante; ein Wert knapp darunter wird verworfen.
    import generative.config as cfg

    # max-cos 0.49 < Default 0.50 → reject; 0.51 → rescue
    plan = _plan("Bildungs-Meta-Analyse-Selektion")
    text = "A meta-analysis selecting randomized controlled trials in education."
    kept_low, rej_low = filter_hallucinated(
        plan, text, semantic_presence_fn=lambda t, x: cfg.TITLE_PRESENCE_COSINE_THRESHOLD - 0.01
    )
    kept_high, rej_high = filter_hallucinated(
        plan, text, semantic_presence_fn=lambda t, x: cfg.TITLE_PRESENCE_COSINE_THRESHOLD + 0.01
    )
    assert kept_low.concepts == [] and rej_low
    assert [c.title for c in kept_high.concepts] == ["Bildungs-Meta-Analyse-Selektion"]
