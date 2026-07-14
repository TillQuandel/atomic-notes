"""db.py — atomic_analytics.db Helper.

Zentrales Modul fuer alle DB-Schreiboperationen der atomic-agent Pipeline.
Dashboard liest direkt per sqlite3, schreibt nie.

Tabellen:
  pipeline_runs — ein Eintrag pro orchestrator.py-Run
  note_evals    — ein Eintrag pro eval_quality_v4.py-Evaluierung

Verwendung:
  from generative.db import get_db, insert_run, insert_eval

  with get_db() as conn:
      insert_run(conn, {...})
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from shared.db_schema import SCHEMA_SQL as _SCHEMA

_REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("ATOMIC_DB_PATH", _REPO_ROOT / ".cache" / "atomic_analytics.db"))


def _add_column(conn: sqlite3.Connection, table: str, coldef: str) -> None:
    """Idempotentes `ALTER TABLE <table> ADD COLUMN <coldef>`.

    #197 Nachbesserung: „duplicate column name" ist der erwartete No-op (Spalte
    existiert bereits). Jeder ANDERE `OperationalError` — insbesondere „database
    is locked" bei Parallel-Prozessen — wird NICHT verschluckt, sondern
    re-raised. Der frühere pauschale `except OperationalError: pass` deutete
    einen Lock-Fehlschlag als „Spalte existiert" fehl: die Spalte fehlte
    anschließend, und spätere Inserts crashten. `busy_timeout` (s. init_db)
    fängt die Race im Normalfall ab; das re-raise ist die Absicherung.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def init_db(path: Path = DB_PATH) -> None:
    """Erstellt DB + Schema falls nicht vorhanden. Idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    # #197 Nachbesserung: bei Parallel-Prozessen kann ein ALTER TABLE an einem
    # Lock scheitern — busy_timeout lässt SQLite bis 5s auf die Freigabe warten,
    # statt sofort mit „database is locked" abzubrechen (kein WAL-Umbau).
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    # Migration für bestehende DBs ohne n_dropped
    _add_column(conn, "pipeline_runs", "n_dropped INT DEFAULT 0")
    # #197 Schritt 2: n_extracted = "nach Planner/Extractor generiert" (Funnel-Top).
    # Bewusst additiv — n_generated (= geschriebene Notes) bleibt unangetastet,
    # damit Alt- und Neu-Zeilen vergleichbar bleiben (keine Migration/Mutation).
    _add_column(conn, "pipeline_runs", "n_extracted INT DEFAULT 0")
    _add_column(conn, "pipeline_runs", "n_words INT DEFAULT 0")
    _add_column(conn, "pipeline_runs", "model TEXT DEFAULT ''")
    _add_column(conn, "pipeline_runs", "cost_usd REAL DEFAULT 0.0")
    # #235: aktives Runtime-Profil (legacy/fast/balanced/quality) mitschreiben —
    # bisher nur im Stdout-Log, dadurch Alt- und A/B-Läufe in der DB ununterscheidbar.
    _add_column(conn, "pipeline_runs", "profile TEXT DEFAULT ''")
    # #239: echte Wall-Clock-Zeit inkl. Stage-8-Eval — siehe SCHEMA_SQL-Kommentar
    # in shared/db_schema.py fuer den Zwei-Phasen-Schreibpfad (insert_run VOR,
    # update_wall_clock_s NACH Stage-8).
    _add_column(conn, "pipeline_runs", "wall_clock_s REAL DEFAULT 0")
    _add_column(conn, "note_evals", "anchor_rate REAL")
    # Anker-Roh-Counts: für die gepoolte Halluzinationsrate (Σ halluziniert /
    # Σ gesamt) im Dashboard — die Pipeline berechnet sie ohnehin, persistiert
    # sie aber bisher nur ins JSONL, nicht in die DB.
    for _col in ("anchors_total", "anchors_hallucinated"):
        _add_column(conn, "note_evals", f"{_col} INT")
    conn.commit()
    conn.close()


@contextmanager
def get_db(path: Path = DB_PATH):
    """Context-Manager: liefert Connection mit WAL + foreign keys.

    Benutze immer `with get_db() as conn:` — commit/rollback automatisch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_run(conn: sqlite3.Connection, data: dict) -> None:
    """Schreibt einen Pipeline-Run in pipeline_runs.

    data-Keys (alle optional ausser run_id):
      run_id, timestamp, pipeline_version, pdf_source, pdf_key, pdf_label,
      n_generated, n_extracted, n_vault, n_inbox, n_merge, n_dropped, n_words,
      model, tokens_total, tokens_input, tokens_output, tokens_cache_read,
      duration_s, eval_version, profile, wall_clock_s

    n_generated = geschriebene Notes (historische Semantik, unangetastet).
    n_extracted = "nach Planner/Extractor generiert" (Funnel-Top, #197). Fehlt
    der Key (Alt-Aufrufer), wird 0 geschrieben — keine NULL, kein Crash.

    wall_clock_s (#239): beim Insert (VOR Stage-8) identisch zu duration_s —
    der Aufrufer (orchestrator.main()) uebergibt hier bewusst denselben Wert.
    Nach Stage-8 korrigiert update_wall_clock_s() die Zeile auf die echte
    Gesamtzeit inkl. Eval-Phase.
    """
    data.setdefault("timestamp", datetime.utcnow().isoformat())
    conn.execute(
        """
        INSERT OR REPLACE INTO pipeline_runs
          (run_id, timestamp, pipeline_version, pdf_source, pdf_key, pdf_label,
           n_generated, n_extracted, n_vault, n_inbox, n_merge, n_dropped, n_words, model,
           cost_usd, tokens_total, tokens_input, tokens_output, tokens_cache_read,
           duration_s, eval_version, fully_cached, profile, wall_clock_s)
        VALUES
          (:run_id, :timestamp, :pipeline_version, :pdf_source, :pdf_key, :pdf_label,
           :n_generated, :n_extracted, :n_vault, :n_inbox, :n_merge, :n_dropped, :n_words, :model,
           :cost_usd, :tokens_total, :tokens_input, :tokens_output, :tokens_cache_read,
           :duration_s, :eval_version, :fully_cached, :profile, :wall_clock_s)
    """,
        {
            "run_id": data.get("run_id"),
            "timestamp": data.get("timestamp"),
            "pipeline_version": data.get("pipeline_version"),
            "pdf_source": data.get("pdf_source"),
            "pdf_key": data.get("pdf_key"),
            "pdf_label": data.get("pdf_label"),
            "n_generated": data.get("n_generated", 0),
            "n_extracted": data.get("n_extracted", 0),
            "n_vault": data.get("n_vault", 0),
            "n_inbox": data.get("n_inbox", 0),
            "n_merge": data.get("n_merge", 0),
            "n_dropped": data.get("n_dropped", 0),
            "n_words": data.get("n_words", 0),
            "model": data.get("model", ""),
            "cost_usd": data.get("cost_usd", 0.0),
            "tokens_total": data.get("tokens_total", 0),
            "tokens_input": data.get("tokens_input", 0),
            "tokens_output": data.get("tokens_output", 0),
            "tokens_cache_read": data.get("tokens_cache_read", 0),
            "duration_s": data.get("duration_s", 0.0),
            "eval_version": data.get("eval_version"),
            "fully_cached": 1 if (data.get("tokens_total", 0) == 0 and data.get("duration_s", 0) > 0) else 0,
            "profile": data.get("profile", ""),
            "wall_clock_s": data.get("wall_clock_s", 0.0),
        },
    )


def update_wall_clock_s(conn: sqlite3.Connection, run_id: str, wall_clock_s: float) -> None:
    """Korrigiert `wall_clock_s` eines bereits per insert_run() geschriebenen Runs (#239).

    insert_run() laeuft VOR Stage-8 und schreibt dort nur die Zeit bis dahin
    (identisch zu duration_s — Stage-8 ist noch nicht gelaufen). Nach Abschluss
    von Stage-8 (Eval-Phase) ruft orchestrator.main() dies auf, um die Zeile auf
    die tatsaechliche Gesamtlaufzeit zu korrigieren. No-op falls run_id nicht
    existiert (kein impliziter Insert hier — insert_run() bleibt der einzige
    Schreibpfad, der eine neue Zeile anlegt).
    """
    conn.execute("UPDATE pipeline_runs SET wall_clock_s = ? WHERE run_id = ?", (wall_clock_s, run_id))


def insert_eval(conn: sqlite3.Connection, data: dict) -> None:
    """Schreibt eine Note-Evaluierung in note_evals.

    data-Keys (alle optional ausser eval_id):
      eval_id, run_id, note_path, acceptance_status,
      hallucination_rate, coverage_factual, coverage_rate,
      tokens_total, tokens_input, tokens_output, tokens_cache_read,
      wall_time_s, pipeline_version, pdf, language, eval_version, timestamp
    """
    data.setdefault("timestamp", datetime.utcnow().isoformat())
    conn.execute(
        """
        INSERT OR REPLACE INTO note_evals
          (eval_id, run_id, note_path, acceptance_status,
           hallucination_rate, anchors_total, anchors_hallucinated,
           coverage_factual, coverage_rate, anchor_rate,
           tokens_total, tokens_input, tokens_output, tokens_cache_read,
           wall_time_s, pipeline_version, pdf, language, eval_version, timestamp)
        VALUES
          (:eval_id, :run_id, :note_path, :acceptance_status,
           :hallucination_rate, :anchors_total, :anchors_hallucinated,
           :coverage_factual, :coverage_rate, :anchor_rate,
           :tokens_total, :tokens_input, :tokens_output, :tokens_cache_read,
           :wall_time_s, :pipeline_version, :pdf, :language, :eval_version, :timestamp)
    """,
        {
            "eval_id": data.get("eval_id"),
            "run_id": data.get("run_id"),
            "note_path": data.get("note_path") or data.get("note"),
            "acceptance_status": data.get("acceptance_status"),
            "hallucination_rate": data.get("hallucination_rate"),
            "anchors_total": data.get("anchors_total"),
            "anchors_hallucinated": data.get("anchors_hallucinated"),
            "coverage_factual": data.get("coverage_factual"),
            "coverage_rate": data.get("coverage_rate"),
            "anchor_rate": data.get("anchor_rate"),
            "tokens_total": data.get("tokens_total"),
            "tokens_input": data.get("tokens_input"),
            "tokens_output": data.get("tokens_output"),
            "tokens_cache_read": data.get("tokens_cache_read"),
            "wall_time_s": data.get("wall_time_s"),
            "pipeline_version": data.get("pipeline_version") or data.get("version"),
            "pdf": data.get("pdf"),
            "language": data.get("language"),
            "eval_version": data.get("eval_version"),
            "timestamp": data.get("timestamp"),
        },
    )


# ---------------------------------------------------------------------------
# Read-Helpers fuer Dashboard
# ---------------------------------------------------------------------------


def query_pipeline_runs(path: Path = DB_PATH) -> list[dict]:
    """Alle pipeline_runs als Liste von Dicts."""
    with get_db(path) as conn:
        rows = conn.execute("SELECT * FROM pipeline_runs ORDER BY timestamp").fetchall()
        return [dict(r) for r in rows]


def query_archived_pipeline_versions(path: Path = DB_PATH) -> list[str]:
    """Versionen archivierter WIP-Läufe (`pipeline_runs_archive`, #193).

    Nur für den Kollisionsschutz des Auto-Version-Bumps gelesen — archivierte
    Versionsnummern bleiben „verbrannt", auch wenn die Läufe aus den
    Dashboard-Quellen ausgelagert sind. Fehlende Tabelle ist kein Fehler."""
    with get_db(path) as conn:
        try:
            rows = conn.execute("SELECT DISTINCT pipeline_version FROM pipeline_runs_archive").fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows if r[0]]


def query_note_evals(
    path: Path = DB_PATH, eval_version: str | None = None, pipeline_version: str | None = None
) -> list[dict]:
    """note_evals mit optionalen Filtern."""
    where, params = [], []
    if eval_version:
        where.append("eval_version = ?")
        params.append(eval_version)
    if pipeline_version:
        where.append("pipeline_version = ?")
        params.append(pipeline_version)
    sql = "SELECT * FROM note_evals"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp"
    with get_db(path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def query_kpi_trend(path: Path = DB_PATH, eval_version: str | None = None) -> dict:
    """KPI-Trend pro Pipeline-Version fuer Dashboard-Sparklines."""
    where = "WHERE eval_version = ?" if eval_version else ""
    params = [eval_version] if eval_version else []
    with get_db(path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                pipeline_version,
                COUNT(*)                          AS n,
                AVG(hallucination_rate)           AS avg_hall,
                AVG(coverage_factual)             AS avg_cov,
                SUM(CASE WHEN acceptance_status='vault' THEN 1 ELSE 0 END) * 100.0
                    / COUNT(*)                    AS accept_rate,
                SUM(tokens_total) / 1e6           AS tokens_m,
                AVG(wall_time_s)  / 60.0          AS avg_dur_min
            FROM note_evals
            {where}
            GROUP BY pipeline_version
            ORDER BY pipeline_version
        """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]


def available_eval_versions(path: Path = DB_PATH) -> list[str]:
    """Alle vorhandenen eval_versions sortiert."""
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT eval_version FROM note_evals WHERE eval_version IS NOT NULL ORDER BY eval_version"
        ).fetchall()
        return [r[0] for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"DB initialisiert: {DB_PATH}")
