"""Tests für #229 (Folge zu #198 P3 / PR #226): total_dur_h-KPI zählte
verwaiste Traces (kein `db_matched`-Filter) weiterhin mit — im Gegensatz zu
`total_tokens`, das seit #226 nur DB-gejointe Läufe summiert.

Wurzelursache: `_calc_kpis` summierte `total_dur_s` ungefiltert über
`token_runs`, während `total_tokens` daneben bereits
`if r.get("db_matched", True)` anwendet. Fix: denselben Filter auf
`total_dur_s` (analog #226).
"""

from __future__ import annotations

from generative.eval_dashboard import _calc_kpis


def _token_run(run_id, duration_min, db_matched=None):
    r = {
        "run_id": run_id,
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_cache": 0,
        "duration_min": duration_min,
        "calls": 1,
    }
    if db_matched is not None:
        r["db_matched"] = db_matched
    return r


def test_orphan_trace_excluded_from_total_dur_h():
    matched = _token_run("r1", 60.0, db_matched=True)  # 1h
    orphan = _token_run("r-orphan", 600.0, db_matched=False)  # 10h, Waise
    kpis = _calc_kpis({}, [], [], [matched, orphan])
    assert kpis["total_dur_h"] == 1.0


def test_legacy_entries_without_db_matched_key_still_count():
    # Standalone-Pfad (main()/_build_html) ruft _calc_kpis ohne Server-Join
    # auf — token_runs tragen dort nie ein db_matched-Feld. Default True.
    legacy = _token_run("r-legacy", 60.0)
    assert "db_matched" not in legacy
    kpis = _calc_kpis({}, [], [], [legacy])
    assert kpis["total_dur_h"] == 1.0


def test_mixed_matched_orphan_and_legacy():
    matched = _token_run("r1", 30.0, db_matched=True)
    orphan = _token_run("r2", 900.0, db_matched=False)
    legacy = _token_run("r3", 30.0)
    kpis = _calc_kpis({}, [], [], [matched, orphan, legacy])
    assert kpis["total_dur_h"] == 1.0
