"""Multi-Perspektiven-Review 2026-07-15, Bug 1 (Datenintegritaets-Fund D2):

Der DB-only-Fallback fuer token_runs (eval_dashboard_server.build_data(),
~Zeile 501 — Laeufe ohne .cache/runs/<run_id>.jsonl-Trace, z. B. extractive)
schrieb tokens_in/tokens_out/tokens_cache hartkodiert auf 0, obwohl
pipeline_runs die Spalten tokens_input/tokens_output/tokens_cache_read
gefuellt hat (duration_min/cost_usd/wall_clock_s wurden im selben Block
bereits korrekt aus der DB gelesen — dasselbe Muster fehlte nur bei den
Token-Feldern). Symptom: v0.3.140 hatte 12.863.095 tokens_total in der DB,
Dashboard zeigte 0/"–" (Totals-Strip Tokens, KPI-Kacheln Billable/ohne-Cache,
kpi_trend.tokens/cost, ch5).

Nebenbefund-Check (Task: "PRUEFE AUCH wall_clock/cur_wall_h im selben
Fallback"): wall_clock_s wurde in diesem Block bereits in #266 korrekt aus
`pipeline_runs.wall_clock_s` gelesen (nicht hartkodiert) — keine zusaetzliche
Luecke. test_db_only_fallback_wall_clock_already_correct ist ein
Regressionsschutz, kein RED->GREEN-Fix.
"""

from __future__ import annotations


def _pipeline_run_db_only(
    run_id="r-extractive",
    ver="v0.3.140",
    tokens_input=8_000_000,
    tokens_output=4_863_095,
    tokens_cache_read=2_000_000,
    wall_clock_s=1800.0,
    duration_s=3600.0,
):
    return {
        "run_id": run_id,
        "timestamp": "2026-07-14T00:00:00",
        "pipeline_version": ver,
        "pdf_source": "Extractive-Quelle - 2020 - Titel.pdf",
        "pdf_key": "extractive-quelle-2020",
        "pdf_label": "Extractive-Quelle - 2020 - Titel",
        "model": "extractive-model",
        "cost_usd": 0.42,
        "n_generated": 3,
        "n_vault": 3,
        "n_inbox": 0,
        "n_merge": 0,
        "n_words": 4000,
        "n_dropped": 0,
        "tokens_total": tokens_input + tokens_output + tokens_cache_read,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_cache_read": tokens_cache_read,
        "duration_s": duration_s,
        "wall_clock_s": wall_clock_s,
    }


def _eval_row(run_id, ver="v0.3.140"):
    return {
        "run_id": run_id,
        "note_path": f"notes/{run_id}.md",
        "acceptance_status": "vault",
        "hallucination_rate": 0.02,
        "coverage_factual": 0.9,
        "pipeline_version": ver,
        "version": ver,
        "pdf": "Extractive-Quelle - 2020 - Titel.pdf",
        "language": "DE→DE",
        "eval_version": "4.1",
        "anchors_total": 5,
        "anchors_hallucinated": 0,
    }


def _build_data_with_db_only_run(monkeypatch, pipeline_run):
    """build_data() mit EINEM pipeline_runs-Eintrag, der KEIN Gegenstueck in
    den JSONL-Traces hat (_read_token_runs() -> []) — genau der DB-only-Fall
    (extractive/aeltere Laeufe ohne Trace-Datei)."""
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    monkeypatch.setattr(_cfg, "AGENT_VERSION", pipeline_run["pipeline_version"])
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [dict(pipeline_run)])
    monkeypatch.setattr(
        _gdb,
        "query_note_evals",
        lambda *a, **k: [_eval_row(pipeline_run["run_id"], pipeline_run["pipeline_version"])],
    )
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])
    return S.build_data()


def test_db_only_fallback_carries_real_token_counts(monkeypatch):
    run = _pipeline_run_db_only()
    data = _build_data_with_db_only_run(monkeypatch, run)
    assert data["kpis"]["total_tokens"] == run["tokens_input"] + run["tokens_output"]
    assert data["kpis"]["cur_tokens_in"] == run["tokens_input"]
    assert data["kpis"]["cur_tokens_out"] == run["tokens_output"]
    assert data["kpis"]["cur_tokens_cache"] == run["tokens_cache_read"]
    assert data["kpis"]["cur_tokens_cache_read"] == run["tokens_cache_read"]


def test_db_only_fallback_feeds_ch5_token_chart(monkeypatch):
    run = _pipeline_run_db_only()
    data = _build_data_with_db_only_run(monkeypatch, run)
    assert run["pipeline_version"] in data["tokens"]["labels"]
    idx = data["tokens"]["labels"].index(run["pipeline_version"])
    assert data["tokens"]["tokens_in"][idx] == run["tokens_input"]
    assert data["tokens"]["tokens_out"][idx] == run["tokens_output"]


def test_db_only_fallback_wall_clock_already_correct(monkeypatch):
    run = _pipeline_run_db_only(wall_clock_s=1800.0)
    data = _build_data_with_db_only_run(monkeypatch, run)
    assert data["kpis"]["cur_wall_h"] == round(1800.0 / 3600, 2)
