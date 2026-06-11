"""Tests für die Dashboard-Nützlichkeits-Features (2026-06-10).

- Kalibrierungs-Arbeitsliste: ungelabelte zuerst, nach LLM-Fehlerquote
  absteigend (None ans Ende), gelabelte danach.
- kpi_vault_n: Vault-Notes der KPI-Version (Basis des Eval-Coverage-Hinweises).
"""
from __future__ import annotations

import sqlite3

import pytest

from generative import db
from generative.eval_dashboard import _calc_kpis
from generative.eval_dashboard_server import _read_calibration_data


def _seed_worklist_db(path):
    db.init_db(path)
    conn = sqlite3.connect(path)

    def eval_row(note, hall, ts):
        conn.execute(
            "INSERT INTO note_evals (run_id, note_path, hallucination_rate, "
            "coverage_factual, coverage_rate, pipeline_version, pdf, "
            "eval_version, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"run-{note}", note, hall, 0.5, 0.5, "v0.3.135", "Bates.pdf", "4.1", ts),
        )

    # Insert-Reihenfolge absichtlich != erwartete Sortierung
    eval_row("vault__b-niedrig.md", 0.10, "2026-06-01")
    eval_row("vault__d-gelabelt.md", 0.30, "2026-06-02")
    eval_row("vault__a-hoch.md", 0.40, "2026-06-03")
    eval_row("vault__c-ohne-rate.md", None, "2026-06-04")

    conn.execute(
        "INSERT INTO calibration_labels (note_path, eval_version, labeled_at, "
        "n_claims, n_supported, n_hallucinated, n_uncertain, human_hall_rate, "
        "llm_hall_rate, agreement_rate) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("vault__d-gelabelt.md", "4.1", "2026-06-10", 10, 7, 3, 0, 0.3, 0.30, 0.9),
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def worklist_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    _seed_worklist_db(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def test_calibration_rows_sorted_as_worklist(worklist_db):
    rows = _read_calibration_data()["rows"]
    assert [r["note"] for r in rows] == [
        "a-hoch",        # ungelabelt, hall 40 %
        "b-niedrig",     # ungelabelt, hall 10 %
        "c-ohne-rate",   # ungelabelt, keine Rate → ans Ende der ungelabelten
        "d-gelabelt",    # gelabelt → ganz unten
    ]


def _log_run(**over):
    base = {
        "key": "bates", "label": "Bates 2017", "ver": "v0.3.135",
        "n_total": 4, "n_vault": 2, "n_merge": 0, "n_inbox": 0,
        "accept_pct": 50.0, "words": 9000, "pages": 12,
    }
    base.update(over)
    return base


def test_calc_kpis_vault_n_counts_only_kpi_version():
    runs = [
        _log_run(ver="v0.1.0", n_total=10, n_vault=5),   # alte Version: zählt nicht
        _log_run(ver="v0.3.135", n_total=4, n_vault=2),
        _log_run(ver="v0.3.135", n_total=3, n_vault=3),
    ]
    kpis = _calc_kpis({}, runs, [], [])
    assert kpis["kpi_vault_n"] == 5      # 2 + 3, nur v0.3.135
    assert kpis["kpi_accept_n"] == 7     # 4 + 3 generiert
    assert kpis["avg_accept"] == pytest.approx(71.4)


def test_calc_kpis_vault_n_zero_without_runs():
    kpis = _calc_kpis({}, [], [], [])
    assert kpis["kpi_vault_n"] == 0
    assert kpis["avg_accept"] is None
