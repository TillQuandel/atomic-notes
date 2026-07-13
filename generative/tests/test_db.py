import sqlite3
import tempfile
import os


def test_pipeline_run_has_cost_usd():
    from generative import db
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db.init_db(Path(path))
        conn = sqlite3.connect(path)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
        conn.close()
        assert "cost_usd" in cols
    finally:
        os.unlink(path)


def test_insert_run_stores_cost_usd():
    from generative import db
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db.init_db(Path(path))
        with db.get_db(Path(path)) as conn:
            db.insert_run(
                conn,
                {
                    "run_id": "test-run-1",
                    "pipeline_version": "v0.0.1",
                    "cost_usd": 0.1234,
                },
            )
        conn2 = sqlite3.connect(path)
        try:
            row = conn2.execute("SELECT cost_usd FROM pipeline_runs WHERE run_id='test-run-1'").fetchone()
        finally:
            conn2.close()
        assert row is not None
        assert abs(row[0] - 0.1234) < 0.0001
    finally:
        os.unlink(path)


# --- #235: Laufzeit-Profil persistieren ------------------------------------


def test_pipeline_runs_has_profile_column(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
    assert "profile" in cols


def test_insert_run_stores_profile(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(
            conn,
            {
                "run_id": "test-run-profile",
                "pipeline_version": "v0.0.1",
                "profile": "balanced",
            },
        )
    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute("SELECT profile FROM pipeline_runs WHERE run_id='test-run-profile'").fetchone()
    finally:
        conn2.close()
    assert row is not None
    assert row[0] == "balanced"


def test_insert_run_profile_defaults_to_empty_string(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(
            conn,
            {
                "run_id": "test-run-no-profile",
                "pipeline_version": "v0.0.1",
            },
        )
    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute("SELECT profile FROM pipeline_runs WHERE run_id='test-run-no-profile'").fetchone()
    finally:
        conn2.close()
    assert row is not None
    assert row[0] == ""
