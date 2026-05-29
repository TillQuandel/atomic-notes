"""Tests für das Stage-6-Crash-Handling (Issue #17).

Kernverhalten: Eine Note, die in Stage 6 crasht, wird NICHT geschrieben (Draft gedroppt),
stattdessen wird ein JSON-Crash-Report abgelegt. Erfolgreiche Notes laufen unberührt durch.
"""
import asyncio
import json
from pathlib import Path

import orchestrator as orch
import agents.base as base
from schemas.atomic_note import AtomicNoteDraft


def _draft(title="Note", body="Body"):
    return AtomicNoteDraft(title=title, body=body, source_anchors=[],
                           related=[], tags=[], synthesis_confidence="low")


# --- _collect_stage6_results: Drop gecrashter Drafts + Report-Write ---

def test_collect_keeps_survivors_drops_crashes(tmp_path):
    good0, good2 = _draft("Good A"), _draft("Good C")
    payload = {
        "title": "Bad B", "step": "verifier", "exception": "RuntimeError: x",
        "traceback": "...", "prompt": "p", "raw_output": "o",
        "draft_body": "b", "phase": "initial", "run_meta": {"run_id": "r"},
    }
    results = [(0, good0), orch._Stage6Failure(1, payload), (2, good2)]
    failed_dir = tmp_path / "failed" / "r"

    survived, crashes = orch._collect_stage6_results(results, failed_dir)

    assert survived == [good0, good2]      # gecrashte Note gedroppt
    assert len(crashes) == 1
    report = failed_dir / "bad-b.json"
    assert report.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["step"] == "verifier"


def test_collect_preserves_original_order(tmp_path):
    d0, d1, d2 = _draft("A"), _draft("B"), _draft("C")
    # results in beliebiger Reihenfolge (gather-Reihenfolge) — Sortierung via idx
    results = [(2, d2), (0, d0), (1, d1)]
    survived, crashes = orch._collect_stage6_results(results, tmp_path / "f")
    assert survived == [d0, d1, d2]
    assert crashes == []


# --- _run_note_pipeline_guarded: Exception → _Stage6Failure mit Payload ---

def test_guarded_converts_crash_to_failure(monkeypatch):
    draft = _draft("My Note", body="Body text")

    def fake_pipeline(*a, **k):
        # Simuliert: letzter LLM-Call war critic, danach Crash
        base._record_call("critic", "the prompt", "raw out", error="boom")
        raise RuntimeError("boom")

    monkeypatch.setattr(orch, "_run_note_pipeline", fake_pipeline)

    res = orch._run_note_pipeline_guarded(3, 5, draft, _run_meta={"run_id": "r1"})

    assert isinstance(res, orch._Stage6Failure)
    assert res.idx == 3
    p = res.payload
    assert p["title"] == "My Note"
    assert p["step"] == "critic"           # aus dem thread-lokalen Call-Record
    assert p["prompt"] == "the prompt"
    assert p["raw_output"] == "raw out"
    assert p["draft_body"] == "Body text"
    assert p["exception"].startswith("RuntimeError") and "boom" in p["exception"]
    assert "Traceback" in p["traceback"]
    assert p["phase"] == "initial"
    assert p["run_meta"] == {"run_id": "r1"}


def test_guarded_passes_through_success(monkeypatch):
    draft = _draft("OK Note")
    monkeypatch.setattr(orch, "_run_note_pipeline", lambda *a, **k: (7, draft))

    res = orch._run_note_pipeline_guarded(7, 8, draft, _run_meta={})

    assert res == (7, draft)               # kein Failure-Wrapping bei Erfolg


def test_guarded_step_unknown_without_call_record(monkeypatch):
    """Crash bevor irgendein LLM-Call lief → step='unknown', kein Müll-Record vom Vorlauf."""
    draft = _draft("Early Crash")

    def fake_pipeline(*a, **k):
        raise ValueError("noch vor dem ersten Call")

    monkeypatch.setattr(orch, "_run_note_pipeline", fake_pipeline)

    res = orch._run_note_pipeline_guarded(0, 1, draft, _run_meta={})

    assert res.payload["step"] == "unknown"
    assert res.payload["prompt"] == ""
    assert res.payload["exception"].startswith("ValueError")


# --- Integrationstest: AK-Beweis durch die echte Verdrahtung ---

def test_process_all_notes_drops_crashed_draft(monkeypatch, tmp_path):
    """Akzeptanzkriterium: eine in Stage 6 crashende Note erscheint NICHT in der
    Rückgabe (wird nicht geschrieben) und produziert einen Crash-Report.
    Mockt nur die per-Note-Pipeline (echte Crashes sind nicht deterministisch erzeugbar);
    die Verdrahtung guarded→gather→collect→return läuft real."""
    d_good, d_bad = _draft("Good Note"), _draft("Bad Note")

    def fake_pipeline(i, n_total, draft, *a, **k):
        if draft.title == "Bad Note":
            base._record_call("verifier", "prompt", "", error="boom")
            raise RuntimeError("backend down")
        return (i, draft)

    monkeypatch.setattr(orch, "_run_note_pipeline", fake_pipeline)
    failed_dir = tmp_path / "failed"

    survived = asyncio.run(orch.process_all_notes_async(
        [d_good, d_bad], existing_concepts={}, concept_links={},
        chunk_map={}, full_text="", acronym_dict={}, concept_map={},
        quality_report=None, pdf_meta={}, source_path=Path("test.pdf"),
        tag_whitelist=[], failed_dir=failed_dir,
    ))

    assert [d.title for d in survived] == ["Good Note"]   # Bad Note gedroppt
    assert (failed_dir / "bad-note.json").exists()
    report = json.loads((failed_dir / "bad-note.json").read_text(encoding="utf-8"))
    assert report["step"] == "verifier"
    assert report["run_meta"]["pdf"] == "test.pdf"
