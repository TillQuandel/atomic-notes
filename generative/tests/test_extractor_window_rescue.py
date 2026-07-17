"""Issue #308: Fenster-Rescue auf Orchestrator-Ebene, wenn ein Konzept nach
Erst-Call UND dem #280-Retry (beide auf demselben 400-Wort-Fenster) weiterhin
leer bleibt.

Root-Cause (#280/#308): der Planner weist Konzepten Chunks zu, deren
Belegstelle im knappen 400-Wort-Fenster nicht ausreicht -- der #297-Retry lief
bisher auf demselben Fenster und verdoppelte nur die Token-Kosten des
Fehlversuchs, ohne die eigentliche Ursache zu beheben. Fix: genau EIN
zusätzlicher Call mit deutlich größerem Fenster (400 -> 1200 Wörter,
`orchestrator._RESCUE_WINDOW_WORDS`), bevor endgültig `dropped`/
`empty_extraction` gebucht wird.
"""

from __future__ import annotations

import asyncio
import json

import generative.agents.tracing as tracing
from generative.agents import extractor
from generative.agents.tracing import JsonlBackend
from generative import orchestrator as orch
from generative.schemas.atomic_note import ConceptItem, ConceptPlan


def _capture(monkeypatch, tmp_path):
    """Biegt das Trace-Backend auf tmp um; gibt einen Reader für stage_outcome-Events zurück.

    Gleiches Muster wie test_stage_outcome_events.py._capture.
    """
    backend = JsonlBackend(run_dir=tmp_path, run_id="test-run")
    monkeypatch.setattr(tracing, "_backend", backend)

    def _read_stage_events() -> list[dict]:
        f = tmp_path / "test-run.jsonl"
        if not f.exists():
            return []
        events = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [e for e in events if e.get("type") == "stage_outcome"]

    return _read_stage_events


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


def _fulltext_with_single_mention(n_filler_words: int = 3000, position: int = 1500) -> str:
    """Baut einen langen Volltext mit genau EINER Erwähnung von 'Amotivation',
    umgeben von generischem Fülltext -- damit `concept_text_window` bei
    window_words=400 tatsächlich ein anderes (kleineres) Fenster liefert als
    bei window_words=1200. Echte Verifikation statt Mock-Attrappe: der Rescue
    muss wirklich mehr Kontext einsammeln, nicht nur behaupten es zu tun."""
    filler = ["Textabschnitt", "mit", "generischem", "Inhalt", "ohne", "Bezug", "zum", "Zielbegriff"]
    words = [filler[i % len(filler)] for i in range(n_filler_words)]
    sentence = (
        "Amotivation beschreibt einen Zustand ohne jegliche Handlungsintention "
        "der bei Deci und Ryan als Gegenpol zur intrinsischen Motivation gilt"
    ).split()
    words[position:position] = sentence  # an fester Position einfügen
    return " ".join(words)


# --- Orchestrator-Ebene: run_extractors_per_concept -------------------------


def test_window_rescue_recovers_after_double_empty(monkeypatch, tmp_path):
    """Erst-Call UND #280-Retry liefern stummes <!--END-->; der #308-Rescue mit
    expandiertem Fenster liefert dann eine Note. Erwartet: genau 3 Calls total
    (kein zusätzlicher interner Retry beim Rescue-Call selbst), Note landet in
    drafts, KEIN dropped-Event, log-Signatur [extractor-window-rescue]."""
    read = _capture(monkeypatch, tmp_path)
    calls = {"n": 0}
    seen_windows: list[int] = []

    real_window = orch.pdf_chunker.concept_text_window

    def spy_window(full_text, search_terms, window_words=400, max_chars=8000):
        seen_windows.append(window_words)
        return real_window(full_text, search_terms, window_words=window_words, max_chars=max_chars)

    monkeypatch.setattr(orch.pdf_chunker, "concept_text_window", spy_window)

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            return "<!--END-->"
        return _NOTE_RESPONSE

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Amotivation")])
    full_text = _fulltext_with_single_mention()

    drafts, concept_map, dropped, failures = asyncio.run(
        orch.run_extractors_per_concept(full_text, plan, existing_concepts={})
    )

    assert calls["n"] == 3, f"Erwartet genau 3 Calls (Erst+#280-Retry+Rescue), waren {calls['n']}"
    assert 400 in seen_windows and orch._RESCUE_WINDOW_WORDS in seen_windows
    assert [d.title for d in drafts] == ["Amotivation"]
    assert dropped == 0
    assert failures == []
    assert "Amotivation" in concept_map

    events = read()
    assert events == []  # Rescue erfolgreich -> kein dropped-Event


def test_window_rescue_still_dropped_if_also_empty(monkeypatch, tmp_path, capsys):
    """Rescue-Call liefert ebenfalls nur <!--END-->: weiterhin dropped/
    empty_extraction, aber genau 3 Calls total (kein vierter Call -- der
    Rescue-Call selbst darf intern NICHT nochmal auf #280 retryen)."""
    read = _capture(monkeypatch, tmp_path)
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        return "<!--END-->"

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Amotivation")])
    full_text = _fulltext_with_single_mention()

    drafts, concept_map, dropped, failures = asyncio.run(
        orch.run_extractors_per_concept(full_text, plan, existing_concepts={})
    )

    assert calls["n"] == 3, f"Erwartet genau 3 Calls total, waren {calls['n']}"
    assert drafts == []
    assert dropped == 1
    assert failures == []

    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Amotivation"
    assert e["stage"] == "extractor"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "empty_extraction"

    err = capsys.readouterr().err
    assert "[extractor-window-rescue-failed]" in err


def test_window_rescue_skips_call_if_window_identical(monkeypatch, tmp_path):
    """Ist das Dokument so kurz, dass window_words=1200 dasselbe Fenster liefert
    wie window_words=400 (ganzer Text passt in beide), wäre ein dritter Call
    garantiert derselbe Fehlschlag -- kein Rescue-Call, sofort dropped."""
    read = _capture(monkeypatch, tmp_path)
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        return "<!--END-->"

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Amotivation")])
    full_text = "Amotivation ist ein kurzer Testtext ohne genug Woerter fuer ein grosses Fenster."

    drafts, concept_map, dropped, failures = asyncio.run(
        orch.run_extractors_per_concept(full_text, plan, existing_concepts={})
    )

    assert calls["n"] == 2, "Kein dritter Call, wenn das expandierte Fenster identisch wäre"
    assert drafts == []
    assert dropped == 1
    events = read()
    assert len(events) == 1
    assert events[0]["drop_reason"] == "empty_extraction"


def test_window_rescue_only_fires_for_none_not_exceptions(monkeypatch, tmp_path):
    """Harte Call-Ausfälle (Exception) sind kein Rescue-Fall -- nur legitime
    Leer-Extraktionen (None nach Erst+#280-Retry). failures bleibt unverändert."""
    read = _capture(monkeypatch, tmp_path)

    async def fake_run_per_concept(concept, concept_text, existing_concepts, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(orch.extractor, "run_per_concept", fake_run_per_concept)

    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Amotivation")])
    full_text = _fulltext_with_single_mention()

    drafts, concept_map, dropped, failures = asyncio.run(
        orch.run_extractors_per_concept(full_text, plan, existing_concepts={})
    )

    assert drafts == []
    assert dropped == 1
    assert failures == [("Amotivation", "boom")]
    events = read()
    assert len(events) == 1
    assert events[0]["drop_reason"] == "call_failed"


# --- Extractor-Ebene: retry_empty=False --------------------------------------


def test_retry_empty_false_skips_internal_retry(monkeypatch):
    """retry_empty=False (#308 Rescue-Call-Kontrakt): bei leerem Erst-Call
    sofort None, OHNE den internen #280-Retry -- Kosten-Deckelung für den
    Orchestrator-Rescue, der selbst schon einen Call mit expandiertem Fenster
    ausführt (genau 1 statt bis zu 2 Zusatz-Calls)."""
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        return "<!--END-->"

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    draft = asyncio.run(
        extractor.run_per_concept(
            concept=_concept(),
            concept_text="Text ohne Substanz.",
            existing_concepts={},
            retry_empty=False,
        )
    )

    assert calls["n"] == 1
    assert draft is None


def test_retry_empty_default_true_unchanged(monkeypatch):
    """Ohne retry_empty (Default True): bestehendes #280-Verhalten unverändert
    -- Regressionsschutz neben test_extractor_empty_end_retry.py."""
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        return "<!--END-->"

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    draft = asyncio.run(
        extractor.run_per_concept(
            concept=_concept(),
            concept_text="Text ohne Substanz.",
            existing_concepts={},
        )
    )

    assert calls["n"] == 2
    assert draft is None
