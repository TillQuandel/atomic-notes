"""#239: Dashboard-KPI "Laufzeit (aktuell)" ist die Summe der Call-Dauern
(Agent-Rechenzeit, mit Overlap bei paralleler Ausfuehrung) — NICHT die echte
Wall-Clock. Seit #239-Fix persistiert `pipeline_runs.wall_clock_s` die echte
Gesamtzeit (siehe test_db.py, test_orchestrator_wall_clock.py). Diese Tests
decken die Dashboard-Seite ab: `_calc_kpis` muss die echte Wall-Clock als
EIGENE, von der Call-Summe unterscheidbare Kennzahl ausweisen, damit das HTML
beide ehrlich getrennt zeigen kann (nicht nur umbenennen).
"""

from __future__ import annotations

from generative.eval_dashboard import _calc_kpis


def _token_run(run_id, duration_min, wall_clock_s=None, ver="v0.3.140", db_matched=True):
    r = {
        "run_id": run_id,
        "ver": ver,
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_cache": 0,
        "duration_min": duration_min,
        "calls": 1,
        "db_matched": db_matched,
    }
    if wall_clock_s is not None:
        r["wall_clock_s"] = wall_clock_s
    return r


def _quality_row(ver="v0.3.140"):
    # _current_version() (Anker fuer "aktuelle" Version, #191) resolved nur
    # gegen quality_rows — ohne mind. 1 Zeile zur Zielversion bleibt
    # latest_pver None und _calc_kpis faellt auf "alle Versionen" zurueck
    # (derselbe Fallback wie bei cur_dur_h/cur_tokens).
    return {"version": ver, "pipeline_version": ver, "timestamp": "2026-07-13T00:00:00"}


def test_cur_wall_h_present_and_distinct_from_cur_dur_h():
    # Call-Dauer-Summe (60 min) > echte Wall-Clock (25 min) — Parallelitaet.
    runs = [_token_run("r1", duration_min=60.0, wall_clock_s=1500.0)]
    kpis = _calc_kpis({}, [], [_quality_row()], runs, current_version="v0.3.140")
    assert kpis["cur_dur_h"] == 1.0
    assert kpis["cur_wall_h"] == round(1500.0 / 3600, 2)
    assert kpis["cur_wall_h"] != kpis["cur_dur_h"]


def test_cur_wall_h_sums_across_runs_of_current_version():
    runs = [
        _token_run("r1", duration_min=30.0, wall_clock_s=900.0),
        _token_run("r2", duration_min=30.0, wall_clock_s=900.0),
        _token_run("r-other-ver", duration_min=999.0, wall_clock_s=999999.0, ver="v0.2.0"),
    ]
    kpis = _calc_kpis({}, [], [_quality_row()], runs, current_version="v0.3.140")
    assert kpis["cur_wall_h"] == round((900.0 + 900.0) / 3600, 2)


def test_cur_wall_h_missing_key_defaults_to_zero():
    # token_runs ohne wall_clock_s-Key (Alt-Traces vor #239 oder Standalone-
    # Pfad ohne Server-Join) duerfen nicht crashen — 0 statt KeyError.
    runs = [_token_run("r1", duration_min=30.0, wall_clock_s=None)]
    kpis = _calc_kpis({}, [], [_quality_row()], runs, current_version="v0.3.140")
    assert kpis["cur_wall_h"] == 0.0


# ── Server-Join: eval_dashboard_server.build_data() traegt wall_clock_s aus ─
# pipeline_runs in token_runs ein (dieselbe Stelle wie pdf_label/ver/cost_usd,
# #198 P3) — Regressionsschutz gegen "KPI-Feld existiert, aber der Join
# vergisst die neue Spalte" (Klasse Fund wie #229 fuer total_dur_h).


def _pipeline_run(run_id, ver="v0.3.140", wall_clock_s=0.0):
    return {
        "run_id": run_id,
        "timestamp": "2026-07-13T00:00:00",
        "pipeline_version": ver,
        "pdf_source": "Testquelle - 2020 - Titel.pdf",
        "pdf_key": "testquelle-2020",
        "pdf_label": "Testquelle - 2020 - Titel",
        "model": "test-model-x",
        "cost_usd": 0.0,
        "n_generated": 1,
        "n_vault": 1,
        "n_inbox": 0,
        "n_merge": 0,
        "n_words": 1000,
        "n_dropped": 0,
        "duration_s": 730.7,
        "wall_clock_s": wall_clock_s,
    }


def test_build_data_join_carries_wall_clock_s_into_kpis(monkeypatch):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    monkeypatch.setattr(_cfg, "AGENT_VERSION", "v0.3.140")
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [_pipeline_run("r1", wall_clock_s=1691.0)])
    monkeypatch.setattr(
        _gdb,
        "query_note_evals",
        lambda *a, **k: [
            {
                "run_id": "r1",
                "note_path": "notes/r1.md",
                "acceptance_status": "vault",
                "hallucination_rate": 0.05,
                "coverage_factual": 0.8,
                "pipeline_version": "v0.3.140",
                "version": "v0.3.140",
                "pdf": "Testquelle - 2020 - Titel.pdf",
                "language": "DE→DE",
                "eval_version": "4.1",
                "anchors_total": 10,
                "anchors_hallucinated": 1,
            }
        ],
    )
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(
        D,
        "_read_token_runs",
        lambda: [
            {"run_id": "r1", "tokens_in": 100, "tokens_out": 50, "tokens_cache": 0, "duration_min": 27.7, "calls": 3}
        ],
    )

    data = S.build_data()
    assert data["kpis"]["cur_wall_h"] == round(1691.0 / 3600, 2)


# ── HTML-Label-Ehrlichkeit: "Laufzeit" ist umbenannt, Wall-Clock ist eine ──
# eigene, sichtbare Kachel (nicht nur ein stiller Datenfeld-Zusatz).


def test_html_separates_agent_compute_time_from_wall_clock():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    # Alte, irrefuehrende Bezeichnung "Laufzeit (aktuell)" (impliziert
    # Wall-Clock, war aber die Call-Dauer-Summe) darf nicht mehr vorkommen.
    assert "label:'Laufzeit (aktuell)'" not in html
    assert "label:'Agent-Rechenzeit (Summe)'" in html
    assert "label:'Wall-Clock (aktuell)'" in html
    assert "kpis.cur_wall_h" in html
