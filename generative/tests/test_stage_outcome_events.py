"""Tests für Stage-Outcome-Events an den Gate-Punkten (#197 Schritt 1+2).

Schritt 1: Ein konsistentes `stage_outcome`-Trace-Event pro Note an jedem
Gate-Punkt, das den Weg durch die Pipeline maschinenlesbar macht (Basis für
den Gate-Funnel, #197 Schritt 3). Deckt die bisher stummen Gates/Drop-Klassen
ab: Artifact-Drop, Stage-6-Crash, Exact-/Sibling-Dedup, Faithfulness. Verifier
und Critic tragen ihr per-Note-Urteil bereits via anchor_stats/score_result.

Schritt 2: neues DB-Feld `n_extracted` ("nach Planner/Extractor generiert"),
das `n_generated` (= geschriebene Notes, Alt-Daten-Kompatibilität) NICHT
antastet.

Backend-Muster wie test_per_agent_tracking.py: echtes JsonlBackend auf tmp
umbiegen, JSONL zurücklesen — nie die produktive .cache/runs treffen.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from unittest.mock import patch

import generative.agents.tracing as tracing
from generative.agents.tracing import JsonlBackend
from generative import orchestrator as orch
from generative.pipeline.claims import Claim
from generative.pipeline.faithfulness_gate import ClaimVerdict, GateResult
from generative.schemas.atomic_note import AtomicNoteDraft, ConceptItem, ConceptPlan
from generative.schemas.citation import CitationMeta


# --- Helfer -----------------------------------------------------------------


def _draft(**kw) -> AtomicNoteDraft:
    base = dict(
        title="T",
        body="b",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="low",
        action="create",
    )
    base.update(kw)
    return AtomicNoteDraft(**base)


def _capture(monkeypatch, tmp_path):
    """Biegt das Trace-Backend auf tmp um; gibt einen Reader für stage_outcome-Events zurück."""
    backend = JsonlBackend(run_dir=tmp_path, run_id="test-run")
    monkeypatch.setattr(tracing, "_backend", backend)

    def _read_stage_events() -> list[dict]:
        f = tmp_path / "test-run.jsonl"
        if not f.exists():
            return []
        events = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [e for e in events if e.get("type") == "stage_outcome"]

    return _read_stage_events


def _citation() -> CitationMeta:
    return CitationMeta(author="Autor", year="2020", title=None, doi=None, source_file="x.pdf")


def _claim(text: str = "Claim-Text (S. 1).") -> Claim:
    return Claim(text=text, anchor_page=1, anchor_span=(0, len(text)), risk_types=["number"], is_quote=False)


# --- Artifact-Drop ----------------------------------------------------------


def test_drop_artifacts_emits_stage_outcome(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    ghost = _draft(title="Geist-Konzept", body="Dieses Konzept wird im Text nicht behandelt.")

    kept = orch._drop_artifacts([ghost])

    assert kept == []
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Geist-Konzept"
    assert e["stage"] == "artifact"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "absence_artifact"


def test_drop_artifacts_no_event_for_kept(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    ok = _draft(title="Echtes Konzept", body="Substanzieller Inhalt mit Beleg (S. 3).")

    kept = orch._drop_artifacts([ok])

    assert [d.title for d in kept] == ["Echtes Konzept"]
    assert read() == []  # kein Drop → kein stage_outcome


# --- Exact-Dedup ------------------------------------------------------------


def test_dedup_exact_emits_stage_outcome(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    a = _draft(title="Gleicher Titel")
    b = _draft(title="Gleicher Titel")  # identischer normalisierter Titel → Dup

    kept = orch.dedup_exact([a, b], {})

    assert len(kept) == 1
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Gleicher Titel"
    assert e["stage"] == "dedup"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "exact_dup"


# --- Sibling-Dedup ----------------------------------------------------------


def test_resolve_sibling_dups_emits_stage_outcome(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    survivor = _draft(title="Alpha", critic_score=4, action="create")
    dropped = _draft(title="Beta", critic_score=2, action="extend", extend_path="Alpha")

    kept, n_sib = orch.resolve_sibling_dups([survivor, dropped], {})

    assert n_sib == 1
    assert [d.title for d in kept] == ["Alpha"]
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Beta"
    assert e["stage"] == "dedup"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "sibling_neardup"
    assert e["detail"] == "Alpha"  # Survivor-Titel als Detail


# --- Stage-6-Crash ----------------------------------------------------------


def test_collect_stage6_results_emits_stage_outcome(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    payload = {
        "title": "Crash-Note",
        "step": "critic",
        "phase": "initial",
        "exception": "RuntimeError: boom",
        "traceback": "...",
        "prompt": "p",
        "raw_output": "o",
        "draft_body": "b",
        "run_meta": {"run_id": "r"},
    }
    results = [(0, _draft(title="Good")), orch._Stage6Failure(1, payload)]

    survived, crashes = orch._collect_stage6_results(results, tmp_path / "failed")

    assert [d.title for d in survived] == ["Good"]
    assert len(crashes) == 1
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Crash-Note"
    assert e["stage"] == "stage6"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "stage6_crash"
    assert e["detail"] == "critic/initial"  # step/phase


# --- Faithfulness -----------------------------------------------------------


def test_faithfulness_downgrade_emits_stage_outcome(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "ENABLE_FAITHFULNESS_GATE", True)
    verdict = ClaimVerdict(claim=_claim(), status="failed_entailment", evidence="Beleg", entailment=0.1)
    fake_gate = GateResult(verdicts=[verdict], failed=True, n_supported=0, n_failed=1, n_abstained=0)
    monkeypatch.setattr(
        "generative.pipeline.faithfulness_gate.run_faithfulness_gate",
        lambda body, page_index, citation, **kw: fake_gate,
    )

    draft = _draft(title="Unbelegt", action="create")
    orch._apply_faithfulness_gate(draft, {1: "text"}, _citation())

    assert draft.faithfulness_fail is True
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Unbelegt"
    assert e["stage"] == "faithfulness"
    assert e["outcome"] == "downgraded"
    assert e["drop_reason"] == "faithfulness_fail"


def test_faithfulness_pass_emits_stage_outcome(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "ENABLE_FAITHFULNESS_GATE", True)
    fake_gate = GateResult(verdicts=[], failed=False, n_supported=2, n_failed=0, n_abstained=0)
    monkeypatch.setattr(
        "generative.pipeline.faithfulness_gate.run_faithfulness_gate",
        lambda body, page_index, citation, **kw: fake_gate,
    )

    draft = _draft(title="Belegt", action="create")
    orch._apply_faithfulness_gate(draft, {1: "text"}, _citation())

    assert draft.faithfulness_fail is False
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Belegt"
    assert e["stage"] == "faithfulness"
    assert e["outcome"] == "passed"
    assert e.get("drop_reason") is None


def test_faithfulness_skip_emits_stage_outcome(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "ENABLE_FAITHFULNESS_GATE", True)
    # Gate darf gar nicht laufen (action != create) — trotzdem ein Skip-Event.
    monkeypatch.setattr(
        "generative.pipeline.faithfulness_gate.run_faithfulness_gate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Gate darf nicht laufen")),
    )

    draft = _draft(title="Merge-Stub", action="extend")
    orch._apply_faithfulness_gate(draft, {1: "text"}, _citation())

    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["stage"] == "faithfulness"
    assert e["outcome"] == "skipped"
    assert e["drop_reason"] == "action_not_create"
    assert e["detail"] == "extend"


def test_faithfulness_disabled_emits_no_event(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "ENABLE_FAITHFULNESS_GATE", False)

    draft = _draft(title="X", action="create")
    orch._apply_faithfulness_gate(draft, {1: "text"}, _citation())

    assert read() == []  # global deaktiviert → kein Gate, kein Event


# --- Nachbesserung #197: bisher stumme Funnel-Lücken ------------------------


def test_entity_resolution_merge_emits_stage_outcome(monkeypatch, tmp_path):
    """Cluster-Merge (entity_resolution) verwirft Nicht-Repräsentanten via
    `consumed`-Set — strukturell derselbe Vorgang wie sibling-Dedup, bekam aber
    kein Event. Der Draft, der aus der finalen Liste verschwindet, muss ein
    stage_outcome tragen (Survivor als Detail)."""
    read = _capture(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "ENABLE_ENTITY_RESOLUTION", True)
    d1 = _draft(title="Red Thread of Information", body="Body Content A")
    d2 = _draft(title="Roter Faden der Information", body="Body Content A")

    with patch("generative.orchestrator.embeddings") as mock_emb:
        mock_emb.embed_title.side_effect = ["emb1", "emb2"]
        mock_emb.cosine.side_effect = [0.95, 0.99]  # Title-accept (Stage 1), Body-Cluster (Stage 2)
        mock_emb.embed_body.side_effect = ["body_emb1", "body_emb2"]
        with patch("generative.orchestrator.canonicalizer.merge_cluster") as mock_merge:
            mock_merge.return_value = d1  # Repräsentant überlebt
            results = asyncio.run(orch.entity_resolution([d1, d2]))

    assert len(results) == 1
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Roter Faden der Information"  # nicht-Repräsentant, verschwindet
    assert e["stage"] == "dedup"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "entity_resolution_merge"
    assert e["detail"] == "Red Thread of Information"  # Survivor-Titel


def test_collect_stage6_baseexception_emits_event_and_report(monkeypatch, tmp_path):
    """Der BaseException-Zweig (z.B. asyncio.CancelledError außerhalb des
    guarded Wrappers) droppte die Note bisher lautlos — kein Event, kein
    Crash-Report. Der Nachbar-Zweig (_Stage6Failure) hat beides; konsistent
    instrumentieren."""
    read = _capture(monkeypatch, tmp_path)
    failed_dir = tmp_path / "failed"
    results = [(0, _draft(title="Good")), asyncio.CancelledError()]

    survived, crashes = orch._collect_stage6_results(results, failed_dir)

    assert [d.title for d in survived] == ["Good"]
    assert crashes == []  # BaseException ist kein _Stage6Failure → nicht in crashes
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["stage"] == "stage6"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "stage6_crash"
    assert e["detail"] == "CancelledError"
    assert list(failed_dir.glob("*.json"))  # Crash-Report geschrieben (diagnostizierbar)


def _concept(title: str) -> ConceptItem:
    return ConceptItem(title=title, priority="high", chapter="Ch", action="create")


def test_run_extractors_empty_ctext_emits_stage_outcome(monkeypatch, tmp_path):
    """Konzept nicht im Volltext gefunden (`if not ctext.strip(): continue`) —
    fiel bisher stumm vor jeder Event-Instrumentierung weg."""
    read = _capture(monkeypatch, tmp_path)
    plan = ConceptPlan(source_title="T", source_summary="S", concepts=[_concept("Xyzzy Plughversion Frobnicate")])
    full_text = "Der schnelle braune Fuchs springt über den faulen Hund."  # kein Titel-Token enthalten

    asyncio.run(orch.run_extractors_per_concept(full_text, plan, existing_concepts={}))

    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Xyzzy Plughversion Frobnicate"
    assert e["stage"] == "extractor"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "empty_extraction"


def test_run_extractors_none_and_exception_emit_stage_outcome(monkeypatch, tmp_path):
    """Extractor-Rückgabe None (leer) bzw. Exception (harter Call-Ausfall)
    verschwanden bisher nur in stderr + dropped-Zähler."""
    read = _capture(monkeypatch, tmp_path)

    async def fake_run_per_concept(concept, concept_text, existing_concepts, **kw):
        if concept.title == "BetaConcept":
            raise RuntimeError("boom")
        return None  # AlphaConcept → leere Extraktion

    monkeypatch.setattr(orch.extractor, "run_per_concept", fake_run_per_concept)

    plan = ConceptPlan(
        source_title="T", source_summary="S", concepts=[_concept("AlphaConcept"), _concept("BetaConcept")]
    )
    full_text = (
        "Einleitung zum Thema AlphaConcept mit weiterem Kontext und Wörtern. "
        "Anschließend Ausführungen zu BetaConcept mit zusätzlichem Fülltext drumherum."
    )

    asyncio.run(orch.run_extractors_per_concept(full_text, plan, existing_concepts={}))

    events = read()
    by_reason = {e["drop_reason"]: e for e in events}
    assert set(by_reason) == {"empty_extraction", "call_failed"}
    assert by_reason["empty_extraction"]["title"] == "AlphaConcept"
    assert by_reason["empty_extraction"]["stage"] == "extractor"
    assert by_reason["call_failed"]["title"] == "BetaConcept"
    assert by_reason["call_failed"]["outcome"] == "dropped"


def test_faithfulness_no_page_index_emits_skip(monkeypatch, tmp_path):
    """Gate aktiv, action=create, aber leerer page_index (PDF ohne [S. N]-Marker):
    griff bisher weder Gate noch Skip-Zweig → gar kein Event."""
    read = _capture(monkeypatch, tmp_path)
    monkeypatch.setattr(orch, "ENABLE_FAITHFULNESS_GATE", True)

    draft = _draft(title="Ohne Index", action="create")
    orch._apply_faithfulness_gate(draft, {}, _citation())  # page_index leer

    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "Ohne Index"
    assert e["stage"] == "faithfulness"
    assert e["outcome"] == "skipped"
    assert e["drop_reason"] == "no_page_index"


# --- Schritt 2: n_extracted-DB-Feld -----------------------------------------


def test_pipeline_runs_has_n_extracted_column(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
    assert "n_extracted" in cols


def test_insert_run_stores_n_extracted_independent_of_n_generated(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(conn, {"run_id": "run-1", "n_extracted": 10, "n_generated": 3})

    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute("SELECT n_extracted, n_generated FROM pipeline_runs WHERE run_id='run-1'").fetchone()
    finally:
        conn2.close()
    assert row == (10, 3)  # zwei unabhängige Felder, kein Semantik-Overlap


def test_insert_run_n_extracted_defaults_zero(tmp_path):
    """Alt-Zeilen-Kompatibilität: fehlt n_extracted im Insert, ist es 0 (kein NULL/Crash)."""
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(conn, {"run_id": "run-legacy", "n_generated": 5})

    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute("SELECT n_extracted FROM pipeline_runs WHERE run_id='run-legacy'").fetchone()
    finally:
        conn2.close()
    assert row == (0,)
