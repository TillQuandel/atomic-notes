"""Tests für #211: Lauf-Filter — nur einen bestimmten Pipeline-Run anzeigen.

Der Versions-Filter poolt alle Runs einer Version; ein einzelner Run
(typisch: der letzte) war nicht isolierbar. Neu: `build_data(run=<run_id>)`
filtert die komplette Kaskade auf diesen Run; `all_runs` liefert die
Dropdown-Optionen (jüngste zuerst, gedeckelt).
"""

from __future__ import annotations

from generative.eval_dashboard_server import _run_options


def _pipeline_run(run_id, ver, ts, pdf_label="Testquelle - 2020 - Titel", cost=0.0):
    return {
        "run_id": run_id,
        "timestamp": ts,
        "pipeline_version": ver,
        "pdf_source": f"{pdf_label}.pdf",
        "pdf_key": "testquelle-2020",
        "pdf_label": pdf_label,
        "model": "test-model-x",
        "cost_usd": cost,
        "n_generated": 4,
        "n_vault": 3,
        "n_inbox": 1,
        "n_merge": 0,
        "n_words": 5000,
        "n_dropped": 0,
        "duration_s": 60.0,
    }


def _eval_row(run_id, ver, note="n1", pdf="Testquelle - 2020 - Titel.pdf"):
    return {
        "run_id": run_id,
        "note_path": f"notes/{run_id}-{note}.md",
        "acceptance_status": "vault",
        "hallucination_rate": 0.05,
        "coverage_factual": 0.8,
        "pipeline_version": ver,
        "version": ver,
        "pdf": pdf,
        "language": "DE→DE",
        "eval_version": "4.1",
        "anchors_total": 10,
        "anchors_hallucinated": 1,
    }


# ── _run_options: Dropdown-Basis ────────────────────────────────────────────


def test_run_options_newest_first_with_label():
    runs = [
        _pipeline_run("r-alt", "v0.3.139", "2026-07-01T10:00:00"),
        _pipeline_run("r-neu", "v0.3.140", "2026-07-11T23:46:56"),
    ]
    opts = _run_options(runs)
    assert [o["id"] for o in opts] == ["r-neu", "r-alt"]
    assert "v0.3.140" in opts[0]["label"]
    assert "Testquelle" in opts[0]["label"]


def test_run_options_time_from_run_id_not_db_timestamp():
    # run_id trägt die lokale STARTzeit (Trace-Namensschema); der DB-timestamp
    # ist UTC vom Lauf-Ende und würde als Anzeige verwirren.
    runs = [_pipeline_run("20260712-002220", "v0.3.140", "2026-07-11T22:36:41")]
    opts = _run_options(runs)
    assert opts[0]["label"].startswith("12.07. 00:22")


def test_run_options_capped_and_skips_missing_ids():
    runs = [_pipeline_run(f"r{i:02d}", "v1", f"2026-07-01T10:00:{i:02d}") for i in range(20)]
    runs.append({"timestamp": "2026-07-11T00:00:00", "pipeline_version": "v1"})  # ohne run_id
    opts = _run_options(runs, limit=15)
    assert len(opts) == 15
    assert opts[0]["id"] == "r19"


# ── build_data-Integration ──────────────────────────────────────────────────


def _patched_build_data(monkeypatch, runs, evals, **kwargs):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    monkeypatch.setattr(_cfg, "AGENT_VERSION", "v0.3.140")
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [dict(r) for r in runs])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: [dict(r) for r in evals])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])
    return S.build_data(**kwargs)


def test_build_data_run_filter_isolates_single_run(monkeypatch):
    runs = [
        _pipeline_run("r-alt", "v0.3.140", "2026-07-01T10:00:00", cost=1.0),
        _pipeline_run("r-neu", "v0.3.140", "2026-07-11T23:46:56", cost=2.0),
    ]
    evals = [
        _eval_row("r-alt", "v0.3.140", "alt-a"),
        _eval_row("r-alt", "v0.3.140", "alt-b"),
        _eval_row("r-neu", "v0.3.140", "neu-a"),
    ]
    data = _patched_build_data(monkeypatch, runs, evals, run="r-neu")

    # Nur der gewählte Run in der Kaskade: 1 Eval-Zeile, 1 Log-Run, Kosten nur r-neu
    assert data["quality_by_version"]["v0.3.140"]["n"] == 1
    assert data["runs_by_version"]["v0.3.140"]["n_runs"] == 1
    cost_by_ver = dict(zip(data["kpi_trend"]["versions"], data["kpi_trend"]["cost"]))
    assert cost_by_ver["v0.3.140"] == 2.0

    # Dropdown-Optionen bleiben ungefiltert (alle Runs anwählbar)
    assert [o["id"] for o in data["all_runs"]] == ["r-neu", "r-alt"]


def test_build_data_run_filter_composes_with_pdf(monkeypatch):
    runs = [
        _pipeline_run("r-a", "v0.3.140", "2026-07-01T10:00:00"),
        _pipeline_run("r-b", "v0.3.140", "2026-07-02T10:00:00", pdf_label="Andere - 2021 - Quelle"),
    ]
    evals = [
        _eval_row("r-a", "v0.3.140"),
        _eval_row("r-b", "v0.3.140", pdf="Andere - 2021 - Quelle.pdf"),
    ]
    # Run r-b (andere Quelle) + PDF-Filter auf die Testquelle → leere Schnittmenge
    data = _patched_build_data(monkeypatch, runs, evals, run="r-b", pdf="testquelle - 2020 - titel")
    assert data["kpis"]["n_notes"] in (0, None)


def test_build_data_without_run_unchanged(monkeypatch):
    runs = [
        _pipeline_run("r-alt", "v0.3.140", "2026-07-01T10:00:00"),
        _pipeline_run("r-neu", "v0.3.140", "2026-07-11T23:46:56"),
    ]
    evals = [_eval_row("r-alt", "v0.3.140"), _eval_row("r-neu", "v0.3.140", "neu-a")]
    data = _patched_build_data(monkeypatch, runs, evals)
    assert data["quality_by_version"]["v0.3.140"]["n"] == 2
    assert data["runs_by_version"]["v0.3.140"]["n_runs"] == 2
