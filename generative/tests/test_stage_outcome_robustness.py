"""Dashboard-/DB-Robustheit gegenüber den neuen stage_outcome-Events (#197 Nachbesserung).

stage_outcome-Events landen im selben JSONL-Trace wie echte LLM-Calls, tragen
aber kein `model`-Feld. Ohne Schutz blähen sie den Call-Zähler des Dashboards
auf (`_read_token_runs`) bzw. lassen den model-Backfill des Servers ins Leere
laufen. Zusätzlich: `db._add_column` darf einen Lock-Fehlschlag nicht als
„Spalte existiert" verschlucken.
"""

from __future__ import annotations

import json
import sqlite3

import pytest


def _write_jsonl(runs_dir, name, records):
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / name).write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


# --- Punkt 5: _read_token_runs zählt nur echte LLM-Calls --------------------


def test_read_token_runs_excludes_event_records(monkeypatch, tmp_path):
    from generative import eval_dashboard as ed

    runs = tmp_path / "runs"
    _write_jsonl(
        runs,
        "20260712-101500.jsonl",
        [
            {"type": "stage_outcome", "stage": "extractor", "outcome": "dropped"},  # kein model
            {"type": "note_outcome", "title": "X"},  # bestehendes Bookkeeping, kein model
            {"model": "haiku", "input_tokens": 100, "output_tokens": 50},
            {"model": "opus", "input_tokens": 200, "output_tokens": 80},
        ],
    )
    monkeypatch.setattr(ed, "RUNS_DIR", runs)

    result = ed._read_token_runs()

    assert len(result) == 1
    assert result[0]["calls"] == 2  # nur die 2 model-Records, nicht die 2 Events
    assert result[0]["tokens_in"] == 300
    assert result[0]["tokens_out"] == 130


def test_read_token_runs_no_phantom_row_for_events_only(monkeypatch, tmp_path):
    from generative import eval_dashboard as ed

    runs = tmp_path / "runs"
    _write_jsonl(
        runs,
        "20260712-101500.jsonl",
        [{"type": "stage_outcome", "stage": "dedup", "outcome": "dropped"} for _ in range(3)],
    )
    monkeypatch.setattr(ed, "RUNS_DIR", runs)

    # Nur Events, 0 echte LLM-Calls → count bleibt 0 → keine 0-Token-Phantomzeile.
    assert ed._read_token_runs() == []


# --- Punkt 7: _add_column verschluckt nur duplicate-column ------------------


def test_add_column_swallows_duplicate_but_reraises_other(tmp_path):
    from generative import db

    path = tmp_path / "t.db"
    db.init_db(path)
    conn = sqlite3.connect(str(path))
    try:
        # Bestehende Spalte → „duplicate column name" → No-op (kein Raise).
        db._add_column(conn, "pipeline_runs", "n_generated INT DEFAULT 0")
        # Neue Spalte → wird angelegt; zweiter Aufruf ist ebenfalls No-op.
        db._add_column(conn, "pipeline_runs", "brandneu_col INT DEFAULT 0")
        db._add_column(conn, "pipeline_runs", "brandneu_col INT DEFAULT 0")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
        assert "brandneu_col" in cols

        # Nicht-duplicate OperationalError (kein solcher Tisch) → re-raise, NICHT verschlucken.
        with pytest.raises(sqlite3.OperationalError):
            db._add_column(conn, "kein_tisch", "x INT")
    finally:
        conn.close()
