"""Tests für das TracingBackend-Interface (Task 1: per-agent tracking)."""

import json
import pytest


def test_trace_event_writes_jsonl(tmp_path, monkeypatch):
    from generative.agents.tracing import JsonlBackend, trace_event
    import generative.agents.tracing as tracing

    backend = JsonlBackend(run_dir=tmp_path, run_id="test-run")
    monkeypatch.setattr(tracing, "_backend", backend)

    trace_event("verifier", "anchor_stats", {"total_in": 5, "confirmed": 4})

    lines = (tmp_path / "test-run.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["type"] == "anchor_stats"
    assert entry["agent"] == "verifier"
    assert entry["total_in"] == 5
    assert entry["confirmed"] == 4
    assert "ts" in entry


def test_trace_run_start_writes_model_config(tmp_path, monkeypatch):
    from generative.agents.tracing import JsonlBackend, trace_run_start
    import generative.agents.tracing as tracing

    backend = JsonlBackend(run_dir=tmp_path, run_id="test-run")
    monkeypatch.setattr(tracing, "_backend", backend)
    monkeypatch.setattr(tracing, "_RUN_ID", "test-run")

    trace_run_start({"planner": "opus", "extractor": "sonnet"})

    lines = (tmp_path / "test-run.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["type"] == "run_start"
    assert entry["model_config"]["extractor"] == "sonnet"
    assert entry["run_id"] == "test-run"


def test_model_config_has_required_keys():
    from generative.config import MODEL_CONFIG

    required = {"planner", "extractor", "verifier", "cross_ref", "critic", "canonicalizer"}
    assert required <= set(MODEL_CONFIG.keys())
    for k, v in MODEL_CONFIG.items():
        assert v, f"MODEL_CONFIG['{k}'] ist leer"


def test_verifier_run_emits_anchor_stats(tmp_path, monkeypatch):
    import generative.agents.tracing as tracing
    import generative.agents.verifier as verifier
    from generative.schemas.atomic_note import AtomicNoteDraft

    backend = __import__("generative.agents.tracing", fromlist=["JsonlBackend"]).JsonlBackend(
        run_dir=tmp_path, run_id="test-run"
    )
    monkeypatch.setattr(tracing, "_backend", backend)

    draft = AtomicNoteDraft(
        title="Test Note",
        body="Kurztext.",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="medium",
    )
    verifier.run(draft, chunk_text="[S. 1] Kurztext. Weiterer Text.")

    trace_file = tmp_path / "test-run.jsonl"
    assert trace_file.exists(), "Kein JSONL geschrieben"
    lines = trace_file.read_text(encoding="utf-8").splitlines()
    events = [__import__("json").loads(l) for l in lines]
    anchor_events = [e for e in events if e.get("type") == "anchor_stats"]
    assert len(anchor_events) == 1
    ev = anchor_events[0]
    assert ev["agent"] == "verifier"
    assert "total_in" in ev
    assert "confirmed" in ev
    assert "confirmation_rate" in ev
    assert ev["total_in"] == 0  # empty source_anchors at start
    assert ev["confirmation_rate"] == 0.0  # no confirmed anchors
    assert ev["confirmed"] == 0


def test_critic_run_emits_score_result(tmp_path, monkeypatch):
    import generative.agents.tracing as tracing
    import generative.agents.critic as critic
    from generative import config as _config
    from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor

    backend = tracing.JsonlBackend(run_dir=tmp_path, run_id="test-run")
    monkeypatch.setattr(tracing, "_backend", backend)
    monkeypatch.setattr(_config, "ENABLE_LLM", False)

    draft = AtomicNoteDraft(
        title="Test Note",
        body="Die Kategorialen Beschreibungen sind zentral. (S. 5) " * 5,
        source_anchors=[TextAnchor(quote="zentral", page="S. 5")],
        related=[],
        tags=[],
        synthesis_confidence="medium",
    )
    critic.run(draft)

    lines = (tmp_path / "test-run.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(l) for l in lines]
    score_events = [e for e in events if e.get("type") == "score_result"]
    assert len(score_events) == 1
    ev = score_events[0]
    assert ev["agent"] == "critic"
    assert isinstance(ev["score"], int)
    assert isinstance(ev["hard_gates_pass"], bool)


def test_eval_agent_stats_aggregates_llm_calls(tmp_path):
    import json

    trace = tmp_path / "run.jsonl"
    entries = [
        {"type": "run_start", "run_id": "r1", "model_config": {"extractor": "opus"}, "ts": "2026-01-01T00:00:00"},
        {
            "agent": "extractor",
            "model": "opus",
            "input_tokens": 1000,
            "output_tokens": 200,
            "duration_ms": 3000,
            "cached": False,
            "error": None,
            "ts": "2026-01-01T00:00:01",
        },
        {
            "agent": "extractor",
            "model": "opus",
            "input_tokens": 1200,
            "output_tokens": 250,
            "duration_ms": 4000,
            "cached": False,
            "error": "Timeout",
            "ts": "2026-01-01T00:00:05",
        },
        {
            "type": "anchor_stats",
            "agent": "verifier",
            "total_in": 5,
            "confirmed": 4,
            "confirmation_rate": 0.8,
            "ts": "2026-01-01T00:00:06",
        },
        {"type": "score_result", "agent": "critic", "score": 4, "hard_gates_pass": True, "ts": "2026-01-01T00:00:07"},
        {
            "type": "note_outcome",
            "agent": "orchestrator",
            "destination": "vault",
            "critic_score": 4,
            "ts": "2026-01-01T00:00:08",
        },
        {
            "type": "plan_stats",
            "agent": "orchestrator",
            "written": 1,
            "vault": 1,
            "inbox": 0,
            "vault_rate": 1.0,
            "ts": "2026-01-01T00:00:09",
        },
    ]
    trace.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    from generative import eval_agent_stats as eas

    stats = eas.aggregate(trace)

    assert stats["extractor"]["calls"] == 2
    assert stats["extractor"]["total_input_tokens"] == 2200
    assert stats["extractor"]["total_output_tokens"] == 450
    assert stats["extractor"]["avg_duration_ms"] == 3500
    assert stats["extractor"]["error_rate"] == pytest.approx(0.5)
    assert stats["verifier"]["avg_confirmation_rate"] == pytest.approx(0.8)
    assert stats["critic"]["avg_score"] == pytest.approx(4.0)
    assert stats["orchestrator"]["vault_rate"] == pytest.approx(1.0)


def test_langfuse_backend_write_does_not_crash_without_package(monkeypatch):
    """LangfuseBackend fällt graceful zurück wenn langfuse nicht installiert."""
    import sys

    monkeypatch.setitem(sys.modules, "langfuse", None)

    # Need to force reimport since module may be cached
    if "generative.agents.langfuse_backend" in sys.modules:
        del sys.modules["generative.agents.langfuse_backend"]

    from generative.agents.langfuse_backend import LangfuseBackend

    backend = LangfuseBackend()
    # Kein Crash erwartet, nur no-op
    backend.write({"type": "run_start", "run_id": "test", "model_config": {}, "ts": "2026-01-01T00:00:00"})
    backend.write(
        {
            "agent": "extractor",
            "input_tokens": 100,
            "output_tokens": 50,
            "duration_ms": 1000,
            "cached": False,
            "ts": "2026-01-01T00:00:01",
        }
    )


def test_aggregate_includes_model_config(tmp_path):
    from generative import eval_quality_v4 as eq

    result = eq._aggregate(
        note_path=tmp_path / "test.md",
        pdf_path=tmp_path / "test.pdf",
        pipeline_version="v0.0.0",
        timestamp="2026-01-01T00:00:00",
        language_pair="de-en",
        chunks=[],
        claim_scores=[],
        llm_meta={},
    )
    assert "model_config" in result
    assert isinstance(result["model_config"], dict)
    assert "extractor" in result["model_config"]


def test_run_totals_includes_eval_entries(tmp_path):
    """run_totals summiert flache Run-Tokens über ALLE Calls — inkl. Stage-8
    eval_quality-Einträge, die nach dem Pipeline-Print in den Trace kommen.
    Cached-Einträge werden ausgeschlossen (matcht orchestrator-Verhalten)."""
    import json

    trace = tmp_path / "run.jsonl"
    entries = [
        {"type": "run_start", "run_id": "r1", "model_config": {}, "ts": "t0"},
        # Pipeline (Stages 1-7)
        {
            "agent": "extractor",
            "model": "sonnet",
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_tokens": 100,
            "cache_creation_tokens": 50,
            "cached": False,
            "ts": "t1",
        },
        {"agent": "critic", "model": "haiku", "input_tokens": 200, "output_tokens": 80, "cached": False, "ts": "t2"},
        # Cached-Call darf NICHT zählen
        {
            "agent": "extractor",
            "model": "sonnet",
            "input_tokens": 9999,
            "output_tokens": 9999,
            "cached": True,
            "ts": "t3",
        },
        # Stage-8 Eval (landet nach dem frühen Print im selben Trace)
        {
            "agent": "eval_quality_v3_primary",
            "model": "sonnet",
            "input_tokens": 300,
            "output_tokens": 400,
            "cached": False,
            "ts": "t4",
        },
        {
            "agent": "eval_quality_v3_audit",
            "model": "sonnet",
            "input_tokens": 150,
            "output_tokens": 120,
            "cached": False,
            "ts": "t5",
        },
    ]
    trace.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    from generative import eval_agent_stats as eas

    totals = eas.run_totals(trace)

    # In: 1000+200+300+150 = 1650 (cached 9999 ausgeschlossen)
    assert totals["input"] == 1650
    # Out: 500+80+400+120 = 1100
    assert totals["output"] == 1100
    assert totals["cache_read"] == 100
    assert totals["cache_create"] == 50
    assert totals["total"] == 1650 + 1100 + 100 + 50
    assert "cost_usd" in totals


def test_run_totals_sums_cost_per_call(tmp_path, monkeypatch):
    """cost_usd summiert die Per-Call-Kosten via config.compute_cost_per_call."""
    import json
    from generative import config

    monkeypatch.setattr(config, "BACKEND", "api")
    monkeypatch.setattr(config, "MODEL_PRICING", {"test-model": {"input": 3.0, "output": 15.0, "cache_read": 0.0}})

    trace = tmp_path / "run.jsonl"
    entries = [
        {
            "agent": "extractor",
            "model": "test-model",
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cached": False,
            "ts": "t1",
        },
        {
            "agent": "eval_quality_v3_primary",
            "model": "test-model",
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "cached": False,
            "ts": "t2",
        },
    ]
    trace.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    from generative import eval_agent_stats as eas

    totals = eas.run_totals(trace)

    # Call 1: 3.0 + 15.0 = 18.0 ; Call 2: 3.0 → Summe 21.0
    assert totals["cost_usd"] == pytest.approx(21.0)


def test_run_totals_tolerates_malformed_line(tmp_path):
    """Eine kaputte Zeile (z.B. null-Token-Feld) darf die Aggregation NICHT
    abbrechen — der Rest wird weiterverarbeitet (wie die alten Inline-Schleifen)."""
    import json

    trace = tmp_path / "run.jsonl"
    entries = [
        {"agent": "extractor", "model": "sonnet", "input_tokens": 100, "output_tokens": 50, "cached": False},
        # null-Feld: e.get("input_tokens", 0) liefert None → würde ti += None crashen
        {"agent": "critic", "model": "haiku", "input_tokens": None, "output_tokens": 20, "cached": False},
        {
            "agent": "eval_quality_v3_primary",
            "model": "sonnet",
            "input_tokens": 300,
            "output_tokens": 80,
            "cached": False,
        },
    ]
    trace.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    from generative import eval_agent_stats as eas

    totals = eas.run_totals(trace)

    # Kaputte Zeile übersprungen, valide summiert: In 100+300=400, Out 50+80=130
    assert totals["input"] == 400
    assert totals["output"] == 130
