"""Tests für #198 P3: total_tokens-KPI zählte verwaiste Traces (keine
pipeline_runs-Zeile) mit — Option A (Maintainer-Entscheidung): Lifetime-Summe
nur über DB-gejointe Läufe.

Wurzelursache: `_calc_kpis` summierte `total_tokens` ungefiltert über
`token_runs` (aus `_read_token_runs()`, reine JSONL-Traces ohne DB-Bezug).
Der Server-Join (`build_data`) reichert `token_runs` zwar mit `ver`/`pdf_label`
aus `pipeline_runs` an, lässt Waisen aber mit Leerstrings drin — die KPI zählte
sie weiter mit (im Gegensatz zum Versions-Chart, der über `ver` filtert).

Fix: Server setzt beim Join `tr["db_matched"] = True/False` explizit (Kriterium:
run_id in pipeline_runs vorhanden). `_calc_kpis` summiert `total_tokens` nur
über `tr.get("db_matched", True)` — Default True erhält das Verhalten des
deprecated Standalone-Pfads (`main()`/`_build_html`), der `_calc_kpis` ohne
Server-Join aufruft.
"""

from __future__ import annotations

from generative.eval_dashboard import _calc_kpis


def _token_run(run_id, tin, tout, db_matched=None):
    r = {
        "run_id": run_id,
        "tokens_in": tin,
        "tokens_out": tout,
        "tokens_cache": 0,
        "duration_min": 1.0,
        "calls": 1,
    }
    if db_matched is not None:
        r["db_matched"] = db_matched
    return r


# ── (a)+(b): _calc_kpis filtert auf db_matched ──────────────────────────────


def test_orphan_trace_excluded_from_total_tokens():
    matched = _token_run("r1", 1000, 500, db_matched=True)
    orphan = _token_run("r-orphan", 2_000_000, 433_376, db_matched=False)
    kpis = _calc_kpis({}, [], [], [matched, orphan])
    assert kpis["total_tokens"] == 1500


def test_legacy_entries_without_db_matched_key_still_count():
    # Standalone-Pfad (main()/_build_html) ruft _calc_kpis ohne Server-Join
    # auf — token_runs tragen dort nie ein db_matched-Feld. Default True.
    legacy = _token_run("r-legacy", 1000, 500)
    assert "db_matched" not in legacy
    kpis = _calc_kpis({}, [], [], [legacy])
    assert kpis["total_tokens"] == 1500


def test_mixed_matched_orphan_and_legacy():
    matched = _token_run("r1", 100, 100, db_matched=True)
    orphan = _token_run("r2", 999_999, 999_999, db_matched=False)
    legacy = _token_run("r3", 50, 50)
    kpis = _calc_kpis({}, [], [], [matched, orphan, legacy])
    assert kpis["total_tokens"] == 300


# ── (c): Server-Join setzt db_matched korrekt ───────────────────────────────


def _pipeline_run(run_id, ver="v0.3.140"):
    return {
        "run_id": run_id,
        "timestamp": "2026-07-01T00:00:00",
        "pipeline_version": ver,
        "pdf_source": "Testquelle - 2020 - Titel.pdf",
        "pdf_key": "testquelle-2020",
        "pdf_label": "Testquelle - 2020 - Titel",
        "model": "test-model-x",
        "cost_usd": 0.0,
        "n_generated": 4,
        "n_vault": 3,
        "n_inbox": 1,
        "n_merge": 0,
        "n_words": 5000,
        "n_dropped": 0,
        "duration_s": 60.0,
    }


def _eval_row(run_id, ver="v0.3.140"):
    return {
        "run_id": run_id,
        "note_path": f"notes/{run_id}.md",
        "acceptance_status": "vault",
        "hallucination_rate": 0.05,
        "coverage_factual": 0.8,
        "pipeline_version": ver,
        "version": ver,
        "pdf": "Testquelle - 2020 - Titel.pdf",
        "language": "DE→DE",
        "eval_version": "4.1",
        "anchors_total": 10,
        "anchors_hallucinated": 1,
    }


def _patched_build_data(monkeypatch, runs, evals, token_runs):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    monkeypatch.setattr(_cfg, "AGENT_VERSION", "v0.3.140")
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [dict(r) for r in runs])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: [dict(r) for r in evals])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [dict(tr) for tr in token_runs])
    return S.build_data()


def test_build_data_join_marks_matched_true(monkeypatch):
    runs = [_pipeline_run("r1")]
    evals = [_eval_row("r1")]
    token_runs = [_token_run("r1", 1000, 500)]
    data = _patched_build_data(monkeypatch, runs, evals, token_runs)
    assert data["kpis"]["total_tokens"] == 1500


def test_build_data_join_marks_orphan_false_and_excludes_from_kpi(monkeypatch):
    # r-orphan hat kein Gegenstück in pipeline_runs (Waisen-Trace, #198).
    runs = [_pipeline_run("r1")]
    evals = [_eval_row("r1")]
    token_runs = [
        _token_run("r1", 1000, 500),
        _token_run("r-orphan", 2_000_000, 433_376),
    ]
    data = _patched_build_data(monkeypatch, runs, evals, token_runs)
    # Nur der DB-gejointe Run zaehlt — die Waise (2.433.376 Tokens) faellt raus.
    assert data["kpis"]["total_tokens"] == 1500
