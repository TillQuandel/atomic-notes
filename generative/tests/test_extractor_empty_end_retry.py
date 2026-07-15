"""Issue #280: stummer `<!--END-->`-Drop bekommt jetzt einen Retry-Pfad.

Befund (Testlauf-Serie 2026-07-14): wenn der Extractor ein zugewiesenes
Konzept im Textfenster nicht ausreichend belegt sieht, antwortet er
prompt-konform nur mit `<!--END-->` (kein `<!--NOTE-->`-Block) -- diese
`None`-Rueckgabe wurde bisher OHNE jeden Retry als `dropped`/`empty_extraction`
verbucht, im Gegensatz zum bestehenden Retry-Mechanismus fuer abgeschnittene
(truncated) Bodies (B2, extractor.py:428-450). Traf Kernkonzepte in 3 von 6
Laeufen (z.B. "Amotivation", ein [high]-priorisiertes SDT-Konzept bei Deci &
Ryan). Fix: derselbe Retry-Gedanke fuer den Empty-Fall -- ein zweiter
Call-Versuch, bevor endgueltig None zurueckgegeben wird.
"""

from __future__ import annotations

import asyncio

from generative.agents import extractor
from generative.schemas.atomic_note import ConceptItem
from generative.schemas.citation import CitationMeta


def _concept(title: str = "Amotivation") -> ConceptItem:
    return ConceptItem(title=title, priority="high", chapter="Kap. 3", action="create")


_NOTE_RESPONSE = """\
<!--NOTE-->
title: Amotivation
aliases: Amotivation, Motivationslosigkeit
tags:
proposed_tags:
synthesis_confidence: low
action: create
extend_path:
<!--BODY-->
# Amotivation: Fehlen jeglicher Handlungsintention

Amotivation beschreibt den Zustand ohne intentionale Handlungssteuerung (S. 12).
<!--ANCHOR-->
page: S. 12
<!--QUOTE-->
Amotivation bezeichnet das voelllige Fehlen von Intentionalitaet.
<!--END-->
"""


def test_silent_end_drop_triggers_retry_and_recovers(monkeypatch):
    """Erster Call liefert stummes `<!--END-->` (kein NOTE-Block) -- bisher
    sofort `None`. Nach dem Fix: ein zweiter Versuch, der hier eine Note liefert."""
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "<!--END-->"
        return _NOTE_RESPONSE

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    draft = asyncio.run(
        extractor.run_per_concept(
            concept=_concept(),
            concept_text="Amotivation kommt im Text vor, aber nur am Rande erwaehnt.",
            existing_concepts={},
            citation=CitationMeta(author="Deci", year="1985", title=None, doi=None, source_file="x.pdf"),
        )
    )

    assert calls["n"] == 2, "Erwarteter Retry nach stummem <!--END--> blieb aus"
    assert draft is not None
    assert draft.title == "Amotivation"


def test_silent_end_drop_still_none_if_retry_also_empty(monkeypatch):
    """Liefert auch der Retry nur `<!--END-->`: weiterhin `None` (kein
    Endlos-Retry) -- aber der Call-Zaehler belegt, dass ein zweiter Versuch
    stattfand (kein stummer Single-Shot-Drop mehr)."""
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        return "<!--END-->"

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    draft = asyncio.run(
        extractor.run_per_concept(
            concept=_concept(),
            concept_text="Kein Bezug zum Konzept im Fenster.",
            existing_concepts={},
        )
    )

    assert calls["n"] == 2
    assert draft is None


def test_refine_loop_does_not_retry_empty_end(monkeypatch):
    """Self-Refine-Aufrufe (`revision_hint` gesetzt) duerfen NICHT retryen --
    sonst Retry-Kaskade im Critic-Loop (dieselbe Design-Entscheidung wie beim
    bestehenden B2-Trunkierungs-Retry-Guard)."""
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        return "<!--END-->"

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    draft = asyncio.run(
        extractor.run_per_concept(
            concept=_concept(),
            concept_text="Text.",
            existing_concepts={},
            revision_hint="Critic-Hinweis",
        )
    )

    assert calls["n"] == 1
    assert draft is None
