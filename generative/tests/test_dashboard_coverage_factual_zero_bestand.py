"""D4-Bestand (Multi-Perspektiven-Dashboard-Review 2026-07-15 + Punkt 5 des
Matrix-Rendering-Fix+Politur-Bündels): `_row_coverage()` (#305-Helper, s.
`_matrix_cell_stats`) fixt das `coverage_factual or coverage_rate`-Anti-
Pattern (verschluckt eine ECHTE 0.0-Coverage als falsy) NUR für die neuen
Matrix-/Paarvergleichs-Funktionen — der `_row_coverage`-Docstring selbst
markiert die Bestands-Stellen mit demselben Muster explizit als "fixt ein
separates Ticket". Dieses Ticket:

  eval_dashboard.py:        _calc_kpis (avg_cov), _calc_pdf_table (cov),
                             _chart_scatter (Legacy-main()-Pfad),
                             _build_quality_chart_data (Legacy-main()-Pfad)
  eval_dashboard_server.py: _read_calibration_data (llm_cov, sqlite3.Row!),
                             quality_by_version-Aggregation (d2["cov"]),
                             _chart_scatter_versioned (aktiver Scatter)

Reale 0%-Coverage-Zeilen existieren (Jockisch-Fälle, s. bestehende Matrix-
Tests) — der Bug zeigt an all diesen Stellen faelschlich `coverage_rate`
(oder verwirft die Zeile) statt der echten 0.0.

Jeder Test unten faellt VOR dem Fix (Stelle nutzt `... or ...`) und besteht
NACH dem Fix (Stelle nutzt `_row_coverage`/dieselbe None-nur-Fallback-Logik).
"""

from __future__ import annotations

import sqlite3

import pytest

from generative import db
from generative.eval_dashboard import (
    _build_quality_chart_data,
    _calc_kpis,
    _calc_pdf_table,
    _chart_scatter,
    _row_coverage,
)
from generative.eval_dashboard_server import _read_calibration_data


# ── _row_coverage: muss jetzt auch sqlite3.Row (nicht nur dict) vertragen ──


def test_row_coverage_accepts_plain_dict_zero_not_swallowed():
    assert _row_coverage({"coverage_factual": 0.0, "coverage_rate": 0.8}) == 0.0


def test_row_coverage_accepts_sqlite_row_zero_not_swallowed():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (coverage_factual REAL, coverage_rate REAL)")
    conn.execute("INSERT INTO t VALUES (0.0, 0.8)")
    row = conn.execute("SELECT * FROM t").fetchone()
    assert _row_coverage(row) == 0.0
    conn.close()


def test_row_coverage_sqlite_row_none_falls_back_to_coverage_rate():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (coverage_factual REAL, coverage_rate REAL)")
    conn.execute("INSERT INTO t VALUES (NULL, 0.8)")
    row = conn.execute("SELECT * FROM t").fetchone()
    assert _row_coverage(row) == 0.8
    conn.close()


# ── eval_dashboard.py: _calc_kpis (avg_cov) ────────────────────────────────


def test_calc_kpis_avg_cov_zero_coverage_factual_not_swallowed():
    rows = [
        {"version": "v1", "hallucination_rate": 0.1, "coverage_factual": 0.0, "coverage_rate": 0.9},
        {"version": "v1", "hallucination_rate": 0.1, "coverage_factual": 0.0, "coverage_rate": 0.9},
    ]
    kpis = _calc_kpis({}, [], rows, [], current_version="v1")
    assert kpis["avg_cov"] == 0.0  # nicht 90.0


# ── eval_dashboard.py: _calc_pdf_table (cov je PDF-Zeile) ──────────────────


def test_calc_pdf_table_cov_zero_coverage_factual_not_swallowed():
    rows = [
        {
            "pdf": "a.pdf",
            "note_path": "n1",
            "version": "v1",
            "hallucination_rate": 0.1,
            "coverage_factual": 0.0,
            "coverage_rate": 0.9,
            "timestamp": "2026-01-01T00:00:00",
        }
    ]
    table = _calc_pdf_table({}, [], rows)
    assert len(table) == 1
    assert table[0]["cov"] == 0.0  # nicht 90.0


# ── eval_dashboard.py: _chart_scatter (Legacy-main()-Pfad) ─────────────────


def test_chart_scatter_zero_coverage_factual_not_swallowed():
    rows = [
        {
            "hallucination_rate": 0.1,
            "coverage_factual": 0.0,
            "coverage_rate": 0.9,
            "note": "n1",
            "pdf": "a.pdf",
        }
    ]
    chart = _chart_scatter(rows)
    assert len(chart["points"]) == 1
    assert chart["points"][0]["y"] == 0.0  # nicht 90.0


# ── eval_dashboard.py: _build_quality_chart_data (Legacy-main()-Pfad) ──────


def test_build_quality_chart_data_cov_field_zero_not_swallowed():
    rows = [
        {
            "hallucination_rate": 0.1,
            "coverage_factual": 0.0,
            "coverage_rate": 0.9,
            "note": "n1",
            "pdf": "a.pdf",
            "version": "v1",
        }
    ]
    out = _build_quality_chart_data(rows)
    # _build_quality_chart_data gibt ein dict mit "rows" (Liste der clean rows) zurueck.
    clean = out["rows"]
    assert len(clean) == 1
    assert clean[0]["cov"] == 0.0  # nicht 90.0


# ── eval_dashboard_server.py: _read_calibration_data (llm_cov, sqlite3.Row) ─


def _seed_db(path):
    db.init_db(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO note_evals (run_id, note_path, hallucination_rate, "
        "coverage_factual, coverage_rate, pipeline_version, pdf, "
        "eval_version, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
        ("run-1", "vault__n1.md", 0.1, 0.0, 0.9, "v0.3.135", "Bates.pdf", "4.1", "2026-06-01"),
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def calib_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    _seed_db(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def test_read_calibration_data_llm_cov_zero_coverage_factual_not_swallowed(calib_db):
    rows = _read_calibration_data()["rows"]
    assert len(rows) == 1
    assert rows[0]["llm_cov"] == 0.0  # nicht 90.0


# ── eval_dashboard_server.py: quality_by_version-Aggregation (kpi_trend.cov) ─


def _dbrow(note, ver, pdf, hall, ts, eval_version="4.1", cov_factual=0.0, cov_rate=0.9, run_id=None):
    return {
        "run_id": run_id or f"r-{note}",
        "note_path": note,
        "pipeline_version": ver,
        "version": ver,
        "hallucination_rate": hall,
        "anchors_total": 10,
        "anchors_hallucinated": 0,
        "coverage_factual": cov_factual,
        "coverage_rate": cov_rate,
        "pdf": pdf,
        "eval_version": eval_version,
        "timestamp": ts,
    }


def _patched_build_data(monkeypatch, evals, current_version="v0.3.144", **kwargs):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    monkeypatch.setattr(_cfg, "AGENT_VERSION", current_version)
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: evals)
    monkeypatch.setattr(_gdb, "query_archived_pipeline_versions", lambda *a, **k: [])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])
    return S.build_data(**kwargs)


def test_quality_by_version_avg_cov_zero_coverage_factual_not_swallowed(monkeypatch):
    evals = [
        _dbrow(f"n{i}", "v0.3.144", "a.pdf", 0.1, f"2026-01-01T00:00:{i:02d}", cov_factual=0.0, cov_rate=0.9)
        for i in range(3)
    ]
    data = _patched_build_data(monkeypatch, evals)
    qbv = data["quality_by_version"]["v0.3.144"]
    assert qbv["avg_cov"] == 0.0  # nicht 90.0
    assert qbv["median_cov"] == 0.0


# ── eval_dashboard_server.py: _chart_scatter_versioned (aktiver Scatter) ───


def test_chart_scatter_versioned_zero_coverage_factual_not_swallowed(monkeypatch):
    evals = [_dbrow("n1", "v0.3.144", "a.pdf", 0.1, "2026-01-01T00:00:00", cov_factual=0.0, cov_rate=0.9)]
    data = _patched_build_data(monkeypatch, evals)
    points = data["scatter"]["points"]
    assert len(points) == 1
    assert points[0]["y"] == 0.0  # nicht 90.0
