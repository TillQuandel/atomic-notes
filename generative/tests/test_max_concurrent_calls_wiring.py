"""#101: `RuntimeConfig.max_concurrent_calls` wird geparst, aber war an beiden
Orchestrator-Semaphore-Stellen (run_extractors_per_concept, process_all_notes_async)
ungenutzt — beide bauten ihr `asyncio.Semaphore` aus der festen Konstante
`config.MAX_CONCURRENT_CALLS`. Wer `ATOMIC_AGENT_MAX_CONCURRENT_CALLS` setzt,
änderte still nichts.

Diese Tests messen die tatsächlich beobachtete Nebenläufigkeit (nicht nur, dass
ein Parameter durchgereicht wird) — Akzeptanzkriterium aus Issue #101.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from pathlib import Path

from generative import orchestrator as orch
from generative.runtime_config import LEGACY, load_runtime_config
from generative.schemas.atomic_note import AtomicNoteDraft, ConceptItem, ConceptPlan


def _concept_plan(n: int) -> tuple[ConceptPlan, str]:
    concepts = [
        ConceptItem(title=f"UniqueConceptTitle{i}", priority="high", chapter="Ch", action="create") for i in range(n)
    ]
    plan = ConceptPlan(source_title="T", source_summary="S", concepts=concepts)
    full_text = "\n\n".join(
        f"Einleitender Fülltext zum Thema UniqueConceptTitle{i} mit zusätzlichem Kontext und mehr Wörtern drumherum."
        for i in range(n)
    )
    return plan, full_text


def _draft(title="Note", body="Body"):
    return AtomicNoteDraft(title=title, body=body, source_anchors=[], related=[], tags=[], synthesis_confidence="low")


# --- run_extractors_per_concept (orchestrator.py Semaphore #1) ---


def test_run_extractors_per_concept_env_limits_effective_parallelism(monkeypatch):
    """ATOMIC_AGENT_MAX_CONCURRENT_CALLS=1 muss die tatsächliche Nebenläufigkeit
    im Extractor-Fan-out auf 1 begrenzen."""
    cfg = load_runtime_config(env={"ATOMIC_AGENT_MAX_CONCURRENT_CALLS": "1"})
    assert cfg.max_concurrent_calls == 1

    state = {"current": 0, "max": 0}
    lock = asyncio.Lock()

    async def fake_run_per_concept(concept, concept_text, existing_concepts, **kwargs):
        async with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.05)
        async with lock:
            state["current"] -= 1
        return None

    monkeypatch.setattr(orch.extractor, "run_per_concept", fake_run_per_concept)

    plan, full_text = _concept_plan(4)
    asyncio.run(
        orch.run_extractors_per_concept(
            full_text,
            plan,
            existing_concepts={},
            max_concurrent_calls=cfg.max_concurrent_calls,
        )
    )

    assert state["max"] == 1


def test_run_extractors_per_concept_preset_value_without_env(monkeypatch):
    """Preset-Wert (nicht ENV) muss ebenfalls greifen — Wert kommt hier aus einem
    per dataclasses.replace() abgeleiteten RuntimeConfig, ohne ATOMIC_AGENT_MAX_CONCURRENT_CALLS
    je gesetzt zu haben."""
    cfg = dataclasses.replace(LEGACY, max_concurrent_calls=2)

    state = {"current": 0, "max": 0}
    lock = asyncio.Lock()

    async def fake_run_per_concept(concept, concept_text, existing_concepts, **kwargs):
        async with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.05)
        async with lock:
            state["current"] -= 1
        return None

    monkeypatch.setattr(orch.extractor, "run_per_concept", fake_run_per_concept)

    plan, full_text = _concept_plan(4)
    asyncio.run(
        orch.run_extractors_per_concept(
            full_text,
            plan,
            existing_concepts={},
            max_concurrent_calls=cfg.max_concurrent_calls,
        )
    )

    assert state["max"] == 2


def test_run_extractors_per_concept_default_is_unbounded_by_one(monkeypatch):
    """Kontrast-Test: ohne max_concurrent_calls-Override bleibt das alte
    Verhalten (Konstante) erhalten — mehr als 1 Task darf gleichzeitig laufen."""
    state = {"current": 0, "max": 0}
    lock = asyncio.Lock()

    async def fake_run_per_concept(concept, concept_text, existing_concepts, **kwargs):
        async with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.05)
        async with lock:
            state["current"] -= 1
        return None

    monkeypatch.setattr(orch.extractor, "run_per_concept", fake_run_per_concept)

    plan, full_text = _concept_plan(4)
    asyncio.run(
        orch.run_extractors_per_concept(
            full_text,
            plan,
            existing_concepts={},
        )
    )

    assert state["max"] > 1


# --- process_all_notes_async (orchestrator.py Semaphore #2) ---


def test_process_all_notes_async_env_limits_effective_parallelism(monkeypatch):
    """Zweite Semaphore-Stelle (Stage-6-Pipeline pro Note) — dieselbe Prüfung."""
    cfg = load_runtime_config(env={"ATOMIC_AGENT_MAX_CONCURRENT_CALLS": "1"})
    assert cfg.max_concurrent_calls == 1

    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    def fake_pipeline(i, n_total, draft, *a, **k):
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return (i, draft)

    monkeypatch.setattr(orch, "_run_note_pipeline", fake_pipeline)

    drafts = [_draft(f"Note {i}") for i in range(4)]
    asyncio.run(
        orch.process_all_notes_async(
            drafts,
            existing_concepts={},
            concept_links={},
            chunk_map={},
            full_text="",
            acronym_dict={},
            concept_map={},
            quality_report=None,
            citation=None,
            source_path=Path("test.pdf"),
            tag_whitelist=[],
            runtime_config=cfg,
        )
    )

    assert state["max"] == 1


def test_process_all_notes_async_legacy_fallback_uses_constant(monkeypatch):
    """runtime_config=None (Legacy-Aufrufpfad) darf nicht crashen und muss auf
    die alte Konstante zurückfallen (> 1 gleichzeitig möglich)."""
    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    def fake_pipeline(i, n_total, draft, *a, **k):
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return (i, draft)

    monkeypatch.setattr(orch, "_run_note_pipeline", fake_pipeline)

    drafts = [_draft(f"Note {i}") for i in range(4)]
    asyncio.run(
        orch.process_all_notes_async(
            drafts,
            existing_concepts={},
            concept_links={},
            chunk_map={},
            full_text="",
            acronym_dict={},
            concept_map={},
            quality_report=None,
            citation=None,
            source_path=Path("test.pdf"),
            tag_whitelist=[],
            runtime_config=None,
        )
    )

    assert state["max"] > 1
