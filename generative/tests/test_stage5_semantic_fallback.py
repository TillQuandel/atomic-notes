"""Issue #127: Stage-5-Volltext-Check ist sprachblind.

`concept_text_window()` (der Stage-5-Skip in `run_extractors_per_concept`) ist
rein lexikalisch. Ein deutscher Planner-Titel auf einer englischen Quelle hat
0 Token-Overlap und wird verworfen -- selbst wenn dasselbe Konzept
`planner.filter_hallucinated`s #66-Rettungsanker (semantische Präsenz) bereits
passiert hat (Ebner/Knowles-Befund: 0 Draft-Notes trotz thematisch passender
Quelle). Fix: vor dem endgültigen [skip] denselben semantischen Rettungskanal
anbieten -- fail-closed, reiner OR-Kanal wie #66.

`semantic_window_fn` ist injizierbar (analog `filter_hallucinated`s
`semantic_presence_fn`) -- Tests laufen ohne echtes Embedding-Modell.
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
    """Gleiches Muster wie test_extractor_window_rescue.py._capture."""
    backend = JsonlBackend(run_dir=tmp_path, run_id="test-run")
    monkeypatch.setattr(tracing, "_backend", backend)

    def _read_stage_events() -> list[dict]:
        f = tmp_path / "test-run.jsonl"
        if not f.exists():
            return []
        events = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [e for e in events if e.get("type") == "stage_outcome"]

    return _read_stage_events


def _concept(title: str) -> ConceptItem:
    return ConceptItem(title=title, priority="high", chapter="Kap. 1", action="create")


_NOTE_RESPONSE = """\
<!--NOTE-->
title: Andragogik
aliases: Andragogik
tags:
proposed_tags:
synthesis_confidence: low
action: create
extend_path:
<!--BODY-->
# Andragogik: Erwachsenenpädagogisches Modell

Andragogik beschreibt selbstgesteuertes Lernen bei Erwachsenen (S. 4).
<!--ANCHOR-->
page: S. 4
<!--QUOTE-->
Andragogy has appeared with increasing frequency in the literature.
<!--END-->
"""


# --- (a) semantisch gerettet: Fallback greift, Extractor-Task entsteht ------


def test_semantic_fallback_rescues_crosslingual_title(monkeypatch, tmp_path):
    """Deutscher Titel, englische Quelle -> lexikalisch leer, semantisch
    gerettet (cosine >= Schwelle) -> Extractor läuft mit dem gelieferten
    Fenster, kein dropped-Event, Log-Signatur [semantic-window-fallback]."""
    read = _capture(monkeypatch, tmp_path)
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        return _NOTE_RESPONSE

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    def fake_semantic_window_fn(full_text, title):
        assert title == "Andragogik"
        return "[S. 4] Andragogy has appeared with increasing frequency in the literature.", 0.75

    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Andragogik")])
    full_text = "This chapter discusses adult learning theory without the German term at all."

    drafts, concept_map, dropped, failures = asyncio.run(
        orch.run_extractors_per_concept(
            full_text, plan, existing_concepts={}, semantic_window_fn=fake_semantic_window_fn
        )
    )

    assert calls["n"] == 1
    assert [d.title for d in drafts] == ["Andragogik"]
    assert dropped == 0
    assert failures == []
    assert "Andragogik" in concept_map
    assert read() == []  # kein dropped-Event


def test_semantic_fallback_log_signature(monkeypatch, tmp_path, capsys):
    """Log-Zeile trägt die neue Rettungs-Signatur inkl. Cosine-Wert."""
    _capture(monkeypatch, tmp_path)

    async def fake_call(prompt, *, model, agent, **kwargs):
        return _NOTE_RESPONSE

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    def fake_semantic_window_fn(full_text, title):
        return "gerettetes Fenster", 0.6789

    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Andragogik")])
    asyncio.run(
        orch.run_extractors_per_concept(
            "irrelevanter Volltext ohne Titel-Token",
            plan,
            existing_concepts={},
            semantic_window_fn=fake_semantic_window_fn,
        )
    )

    err = capsys.readouterr().err
    assert "[semantic-window-fallback]" in err
    assert "Andragogik" in err
    assert "cosine=0.68" in err  # :.2f-Formatierung


# --- (b) themenfremd: weiterhin skip + Funnel-Event -------------------------


def test_semantic_fallback_rejects_unrelated_title(monkeypatch, tmp_path, capsys):
    """Titel ohne semantischen Bezug -> Fallback liefert leeres Fenster ->
    weiterhin [skip] + dropped/empty_extraction-Event, kein Extractor-Call."""
    read = _capture(monkeypatch, tmp_path)
    calls = {"n": 0}

    async def fake_call(prompt, *, model, agent, **kwargs):
        calls["n"] += 1
        return _NOTE_RESPONSE

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    def fake_semantic_window_fn(full_text, title):
        return "", 0.12  # unter der Schwelle -> kein Rettungsfenster

    plan = ConceptPlan(
        source_title="T", source_summary="S", concepts=[_concept("Quantenverschränkung in Photonenpaaren")]
    )
    full_text = "Der schnelle braune Fuchs springt über den faulen Hund."

    drafts, concept_map, dropped, failures = asyncio.run(
        orch.run_extractors_per_concept(
            full_text, plan, existing_concepts={}, semantic_window_fn=fake_semantic_window_fn
        )
    )

    assert calls["n"] == 0  # kein Extractor-Call -- schon vorher gedroppt
    assert drafts == []
    # dropped = len(tasks) - len(drafts) (siehe Docstring) -- ein Konzept, das
    # schon VOR dem Task-Dispatch (weder lexikalisch noch semantisch im
    # Volltext) verworfen wird, taucht nie in `tasks` auf und zählt daher
    # NICHT in `dropped` (unverändertes Verhalten des bestehenden [skip]-Pfads,
    # vgl. test_run_extractors_empty_ctext_emits_stage_outcome).
    assert dropped == 0
    assert failures == []

    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Quantenverschränkung in Photonenpaaren"
    assert e["stage"] == "extractor"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "empty_extraction"
    assert e["detail"] == "not in fulltext"

    err = capsys.readouterr().err
    assert "[skip]" in err
    assert "[semantic-window-fallback]" not in err


# --- (c) lexikalischer Treffer: Fallback wird gar nicht erst berechnet -----


def test_semantic_fallback_not_consulted_when_lexically_present(monkeypatch, tmp_path):
    """Titel lexikalisch im Text -> concept_text_window() liefert bereits ein
    Fenster -> semantic_window_fn darf gar nicht erst aufgerufen werden
    (Performance: kein Embedding-Call im Normalfall, vgl. #66-Analogtest
    test_lexically_present_kept_without_consulting_semantic)."""
    calls: list[str] = []

    async def fake_call(prompt, *, model, agent, **kwargs):
        return _NOTE_RESPONSE

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)

    def spy_semantic_window_fn(full_text, title):
        calls.append(title)
        return "sollte nie verwendet werden", 0.99

    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Andragogik")])
    full_text = "Ein Kapitel über Andragogik und ihre Prinzipien beim Lernen Erwachsener."

    drafts, concept_map, dropped, failures = asyncio.run(
        orch.run_extractors_per_concept(
            full_text, plan, existing_concepts={}, semantic_window_fn=spy_semantic_window_fn
        )
    )

    assert calls == []  # lexikalischer Treffer -> kein Fallback-Aufruf
    assert dropped == 0
    assert [d.title for d in drafts] == ["Andragogik"]


# --- Default-Verdrahtung: ohne Injection zeigt der Default auf pdf_chunker --


def test_default_semantic_window_fn_wired_to_pdf_chunker(monkeypatch, tmp_path):
    """Ohne explizite Injection nutzt run_extractors_per_concept
    `pdf_chunker.semantic_concept_window` mit der config-Schwelle
    TITLE_PRESENCE_COSINE_THRESHOLD -- verifiziert per Spy statt echtem
    Modell-Load (deterministisch, kein ML-Dependency in diesem Test)."""
    read = _capture(monkeypatch, tmp_path)
    seen: list[tuple] = []

    def spy_semantic_concept_window(full_text, title, threshold, **kwargs):
        seen.append((title, threshold))
        return "", 0.0

    monkeypatch.setattr(orch.pdf_chunker, "semantic_concept_window", spy_semantic_concept_window)

    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Xyzzy Plughversion")])
    full_text = "Ein völlig anderer Text ohne jeden Bezug."

    asyncio.run(orch.run_extractors_per_concept(full_text, plan, existing_concepts={}))

    assert len(seen) == 1
    title, threshold = seen[0]
    assert title == "Xyzzy Plughversion"
    from generative.config import TITLE_PRESENCE_COSINE_THRESHOLD

    assert threshold == TITLE_PRESENCE_COSINE_THRESHOLD
    events = read()
    assert len(events) == 1 and events[0]["drop_reason"] == "empty_extraction"
