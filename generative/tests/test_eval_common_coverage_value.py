"""#316: `.get("coverage_factual", r.get("coverage_rate", 0))` faellt bei einem
explizit gespeicherten `None`-Wert NICHT auf `coverage_rate` zurueck --
`dict.get`s Default greift nur bei fehlendem Key, nicht bei vorhandenem Key mit
`None`-Wert. `eval_dashboard.py::_row_coverage` fixt genau das bereits fuer die
Dashboard-Stellen (separates Ticket) -- dieser Test deckt den geteilten Helper
fuer die Bestands-Stellen AUSSERHALB des Dashboards ab (eval_progress.py,
orchestrator.py, db.py::query_kpi_trend).

RED vor dem Fix: `coverage_value` existiert nicht in eval_common.py.
"""

from __future__ import annotations

import sqlite3

from generative.eval_common import coverage_value


def test_coverage_value_real_zero_not_swallowed():
    assert coverage_value({"coverage_factual": 0.0, "coverage_rate": 0.9}) == 0.0


def test_coverage_value_none_falls_back_to_coverage_rate():
    assert coverage_value({"coverage_factual": None, "coverage_rate": 0.7}) == 0.7


def test_coverage_value_missing_key_falls_back_to_coverage_rate():
    assert coverage_value({"coverage_rate": 0.7}) == 0.7


def test_coverage_value_both_missing_returns_none():
    assert coverage_value({}) is None


def test_coverage_value_accepts_sqlite_row():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (coverage_factual REAL, coverage_rate REAL)")
    conn.execute("INSERT INTO t VALUES (NULL, 0.8)")
    row = conn.execute("SELECT * FROM t").fetchone()
    assert coverage_value(row) == 0.8
    conn.close()
