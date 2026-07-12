"""Test: run_eval.py persistiert die Anker-Roh-Counts (#196 P4).

Befund (Review): `run_eval.py` schrieb hallucination_rate, aber NICHT
anchors_total/anchors_hallucinated in note_evals — obwohl `eval_note()` sie
liefert und das Dashboard sie für die gepoolte Fehlerquote + Wilson-CI braucht.
Der kanonische Pipeline-Pfad (`eval_quality_v4.save_result`) schreibt sie
bereits; run_eval.py muss konsistent sein.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path


def _fake_result() -> dict:
    return {
        "hallucination_rate": 0.1,
        "anchors_total": 20,
        "anchors_hallucinated": 2,
        "coverage_factual": 0.8,
        "coverage_rate": 0.75,
        "language": "de-de",
        "timestamp": "2026-07-12T00:00:00",
    }


def test_eval_insert_data_maps_anchor_counts():
    import run_eval

    data = run_eval._eval_insert_data(
        _fake_result(),
        eval_id="run1__note",
        run_id="run1",
        note_name="note.md",
        pipeline_version="v0.9.9",
        pdf_name="x.pdf",
    )
    assert data["anchors_total"] == 20
    assert data["anchors_hallucinated"] == 2


def test_run_eval_insert_persists_anchor_counts_to_db():
    from generative import db
    import run_eval

    data = run_eval._eval_insert_data(
        _fake_result(),
        eval_id="run1__note",
        run_id="run1",
        note_name="note.md",
        pipeline_version="v0.9.9",
        pdf_name="x.pdf",
    )
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db.init_db(Path(path))
        with db.get_db(Path(path)) as conn:
            db.insert_run(conn, {"run_id": "run1", "pipeline_version": "v0.9.9"})
            db.insert_eval(conn, data)
        conn2 = sqlite3.connect(path)
        try:
            row = conn2.execute(
                "SELECT anchors_total, anchors_hallucinated FROM note_evals WHERE eval_id='run1__note'"
            ).fetchone()
        finally:
            conn2.close()
        assert row == (20, 2)
    finally:
        os.unlink(path)
