import re
import sqlite3
import tempfile
import os

import pytest


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


# --- #239: echte Wall-Clock (inkl. Stage-8) additiv persistieren -----------
#
# duration_s wird VOR Stage-8 geschrieben (orchestrator.py, insert_run-Call)
# und ist deshalb systematisch zu niedrig (Eval-Phase fehlt, #239-Befund
# 30-37% Untererfassung). wall_clock_s ist eine additive Spalte: beim Insert
# zunaechst identisch zu duration_s (kein Stage-8 passiert noch), nach
# Abschluss von Stage-8 per update_wall_clock_s() auf die echte Gesamtzeit
# korrigiert (siehe test_orchestrator_wall_clock.py fuer den main()-Roundtrip).


def test_pipeline_runs_has_wall_clock_s_column(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
    assert "wall_clock_s" in cols


def test_insert_run_stores_wall_clock_s(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(
            conn,
            {
                "run_id": "test-run-wall-clock",
                "pipeline_version": "v0.0.1",
                "duration_s": 100.0,
                "wall_clock_s": 100.0,
            },
        )
    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute("SELECT wall_clock_s FROM pipeline_runs WHERE run_id='test-run-wall-clock'").fetchone()
    finally:
        conn2.close()
    assert row is not None
    assert row[0] == pytest.approx(100.0)


def test_insert_run_wall_clock_s_defaults_to_zero(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(
            conn,
            {
                "run_id": "test-run-no-wall-clock",
                "pipeline_version": "v0.0.1",
            },
        )
    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute("SELECT wall_clock_s FROM pipeline_runs WHERE run_id='test-run-no-wall-clock'").fetchone()
    finally:
        conn2.close()
    assert row is not None
    assert row[0] == pytest.approx(0.0)


def test_update_wall_clock_s_corrects_existing_row_after_stage8(tmp_path):
    """Simuliert den Zwei-Phasen-Schreibpfad: insert_run() VOR Stage-8 (wall_clock_s
    == duration_s, kein Eval passiert), update_wall_clock_s() NACH Stage-8 (echte
    Gesamtzeit inkl. Eval-Phase, immer >= duration_s)."""
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(
            conn,
            {
                "run_id": "test-run-stage8",
                "pipeline_version": "v0.0.1",
                "duration_s": 1182.4,
                "wall_clock_s": 1182.4,
            },
        )
        db.update_wall_clock_s(conn, "test-run-stage8", 1691.0)
    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute(
            "SELECT duration_s, wall_clock_s FROM pipeline_runs WHERE run_id='test-run-stage8'"
        ).fetchone()
    finally:
        conn2.close()
    assert row is not None
    duration_s, wall_clock_s = row
    assert duration_s == pytest.approx(1182.4)
    assert wall_clock_s == pytest.approx(1691.0)
    assert wall_clock_s > duration_s


def test_update_wall_clock_s_on_unknown_run_id_is_noop(tmp_path):
    """Kein Insert versteckt in update_wall_clock_s — unbekannte run_id aendert nichts."""
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.update_wall_clock_s(conn, "does-not-exist", 42.0)
    conn2 = sqlite3.connect(str(path))
    try:
        rows = conn2.execute("SELECT * FROM pipeline_runs").fetchall()
    finally:
        conn2.close()
    assert rows == []


def test_orchestrator_migrates_existing_db_missing_wall_clock_s_column(tmp_path):
    """Migration wie #235 profile: bestehende DB ohne wall_clock_s bekommt die
    Spalte per init_db()-Aufruf nachgezogen (additiv, kein Datenverlust)."""
    from generative import db
    from shared.db_schema import SCHEMA_SQL

    # Regex statt starrem String-Replace: robust gegen Kommentar-Textaenderungen
    # an der wall_clock_s-Spaltendefinition — matcht Komma + optionale
    # Kommentarzeilen + die Spaltenzeile selbst, unabhaengig vom genauen Wortlaut.
    # #330: wall_clock_s ist seither NICHT mehr die letzte Spalte (abort_reason
    # folgt) — hat also selbst ein trailing Komma (","?" faengt beide Faelle),
    # und die Ersetzung muss der VORHERIGEN Spalte ("profile") ihr eigenes
    # trailing Komma zurueckgeben (",\n" statt nur "\n"), sonst fehlt das
    # Trennzeichen zu abort_reason und das simulierte Alt-Schema ist ungueltiges SQL.
    old_schema = re.sub(r",\n(\s*--[^\n]*\n)*\s*wall_clock_s\s+REAL DEFAULT 0,?\n", ",\n", SCHEMA_SQL)
    assert "wall_clock_s" not in old_schema, "Test-Fixture-Bug: Alt-Schema hat wall_clock_s schon"

    path = tmp_path / "old-schema.db"
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(old_schema)
        conn.commit()
    finally:
        conn.close()

    db.init_db(path)
    conn2 = sqlite3.connect(str(path))
    try:
        cols = [r[1] for r in conn2.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
    finally:
        conn2.close()
    assert "wall_clock_s" in cols


# --- #330: Abbruchgrund fuer 0-Notes-Laeufe additiv persistieren -----------
#
# 0-Notes-Laeufe (Konzeptmangel oder Totalverlust) hinterliessen bisher KEINE
# pipeline_runs-Zeile. abort_reason ist die additive, nullable Spalte, die den
# Grund festhaelt, wenn orchestrator.main() so einen Lauf jetzt trotzdem
# persistiert (n_generated=0). Erfolgspfad-Zeilen bleiben NULL.


def test_pipeline_runs_has_abort_reason_column(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
    assert "abort_reason" in cols


def test_insert_run_stores_abort_reason(tmp_path):
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(
            conn,
            {
                "run_id": "test-run-zero-notes",
                "pipeline_version": "v0.0.1",
                "n_generated": 0,
                "abort_reason": "no_concepts",
            },
        )
    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute("SELECT abort_reason FROM pipeline_runs WHERE run_id='test-run-zero-notes'").fetchone()
    finally:
        conn2.close()
    assert row is not None
    assert row[0] == "no_concepts"


def test_insert_run_abort_reason_defaults_to_null(tmp_path):
    """Erfolgspfad-Aufrufer uebergeben keinen abort_reason-Key -> muss NULL
    bleiben, nicht leerer String (unterscheidbar von einem echten Abbruchgrund)."""
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(
            conn,
            {
                "run_id": "test-run-success",
                "pipeline_version": "v0.0.1",
                "n_generated": 3,
            },
        )
    conn2 = sqlite3.connect(str(path))
    try:
        row = conn2.execute("SELECT abort_reason FROM pipeline_runs WHERE run_id='test-run-success'").fetchone()
    finally:
        conn2.close()
    assert row is not None
    assert row[0] is None


def test_orchestrator_migrates_existing_db_missing_abort_reason_column(tmp_path):
    """Migration wie #235/#239: bestehende DB ohne abort_reason bekommt die
    Spalte per init_db()-Aufruf nachgezogen (additiv, Bestandszeilen bleiben NULL)."""
    from generative import db
    from shared.db_schema import SCHEMA_SQL

    old_schema = re.sub(r",\n(\s*--[^\n]*\n)*\s*abort_reason\s+TEXT\n", "\n", SCHEMA_SQL)
    assert "abort_reason" not in old_schema, "Test-Fixture-Bug: Alt-Schema hat abort_reason schon"

    path = tmp_path / "old-schema.db"
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(old_schema)
        conn.commit()
    finally:
        conn.close()

    db.init_db(path)
    conn2 = sqlite3.connect(str(path))
    try:
        cols = [r[1] for r in conn2.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
    finally:
        conn2.close()
    assert "abort_reason" in cols


def test_query_kpi_trend_avg_cov_falls_back_to_coverage_rate_when_factual_null(tmp_path):
    """#316: `AVG(coverage_factual)` ohne Fallback liefert NULL fuer avg_cov, sobald
    `coverage_factual` NULL ist -- strukturell der Fall fuer JEDE eval_version=4.x-
    Zeile (#233, coverage_factual seit v4 abgeschafft). Fix: `COALESCE(coverage_factual,
    coverage_rate)`.

    RED vor dem Fix: avg_cov ist None. GREEN: avg_cov == 0.7 (coverage_rate)."""
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(conn, {"run_id": "r1", "pipeline_version": "v1"})
        db.insert_eval(
            conn,
            {
                "eval_id": "e1",
                "run_id": "r1",
                "note_path": "n1",
                "acceptance_status": "vault",
                "hallucination_rate": 0.1,
                "coverage_factual": None,
                "coverage_rate": 0.7,
                "tokens_total": 100,
                "wall_time_s": 1.0,
                "pipeline_version": "v1",
                "pdf": "a.pdf",
                "eval_version": "4.1",
            },
        )

    trend = db.query_kpi_trend(path)
    assert len(trend) == 1
    assert trend[0]["avg_cov"] == 0.7


def test_query_kpi_trend_avg_cov_real_zero_coverage_factual_not_swallowed(tmp_path):
    """COALESCE darf eine echte 0.0 in coverage_factual nicht durch coverage_rate
    ersetzen -- COALESCE greift NUR bei SQL-NULL, nicht bei 0.0 (anders als ein
    Python `or`-Fallback), also bereits korrekt -- Regressionsschutz."""
    from generative import db

    path = tmp_path / "test.db"
    db.init_db(path)
    with db.get_db(path) as conn:
        db.insert_run(conn, {"run_id": "r1", "pipeline_version": "v1"})
        db.insert_eval(
            conn,
            {
                "eval_id": "e1",
                "run_id": "r1",
                "note_path": "n1",
                "acceptance_status": "vault",
                "hallucination_rate": 0.1,
                "coverage_factual": 0.0,
                "coverage_rate": 0.9,
                "tokens_total": 100,
                "wall_time_s": 1.0,
                "pipeline_version": "v1",
                "pdf": "a.pdf",
                "eval_version": "1.3",
            },
        )

    trend = db.query_kpi_trend(path)
    assert trend[0]["avg_cov"] == 0.0
