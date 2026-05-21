"""db.py — atomic_analytics.db Helper.

Zentrales Modul fuer alle DB-Schreiboperationen der atomic-agent Pipeline.
Dashboard liest direkt per sqlite3, schreibt nie.

Tabellen:
  pipeline_runs — ein Eintrag pro orchestrator.py-Run
  note_evals    — ein Eintrag pro eval_quality_v4.py-Evaluierung

Verwendung:
  from db import get_db, insert_run, insert_eval

  with get_db() as conn:
      insert_run(conn, {...})
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / ".cache" / "atomic_analytics.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id            TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    pipeline_version  TEXT,
    pdf_source        TEXT,
    pdf_key           TEXT,   -- normaliserter Schlüssel z.B. "bates"
    pdf_label         TEXT,   -- lesbarer Name z.B. "Bates"
    n_generated       INT  DEFAULT 0,
    n_vault           INT  DEFAULT 0,
    n_inbox           INT  DEFAULT 0,
    n_merge           INT  DEFAULT 0,
    n_dropped         INT  DEFAULT 0,
    n_words           INT  DEFAULT 0,
    model             TEXT DEFAULT '',
    cost_usd          REAL DEFAULT 0.0,
    tokens_total      INT  DEFAULT 0,
    tokens_input      INT  DEFAULT 0,
    tokens_output     INT  DEFAULT 0,
    tokens_cache_read INT  DEFAULT 0,
    duration_s        REAL DEFAULT 0,
    eval_version      TEXT,
    fully_cached      INT  DEFAULT 0   -- 1 wenn alle Agent-Calls aus lokalem Cache
);

CREATE TABLE IF NOT EXISTS note_evals (
    eval_id             TEXT PRIMARY KEY,
    run_id              TEXT REFERENCES pipeline_runs(run_id),
    note_path           TEXT,
    acceptance_status   TEXT,
    hallucination_rate  REAL,
    coverage_factual    REAL,
    coverage_rate       REAL,
    anchor_rate         REAL,
    tokens_total        INT,
    tokens_input        INT,
    tokens_output       INT,
    tokens_cache_read   INT,
    wall_time_s         REAL,
    pipeline_version    TEXT,
    pdf                 TEXT,
    language            TEXT,
    eval_version        TEXT,
    timestamp           TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_version   ON pipeline_runs(pipeline_version);
CREATE INDEX IF NOT EXISTS idx_runs_pdf       ON pipeline_runs(pdf_source);
CREATE INDEX IF NOT EXISTS idx_evals_run      ON note_evals(run_id);
CREATE INDEX IF NOT EXISTS idx_evals_version  ON note_evals(pipeline_version);
CREATE INDEX IF NOT EXISTS idx_evals_ev       ON note_evals(eval_version);

-- Kalibrierungs-Labels (manuell, pro Note aggregiert)
CREATE TABLE IF NOT EXISTS calibration_labels (
    note_path           TEXT NOT NULL,
    eval_version        TEXT NOT NULL DEFAULT '4.1',  -- gegen welche LLM-Version verglichen
    labeled_at          TEXT,
    n_claims            INT  DEFAULT 0,
    n_supported         INT  DEFAULT 0,   -- human: s
    n_hallucinated      INT  DEFAULT 0,   -- human: h
    n_uncertain         INT  DEFAULT 0,   -- human: ?
    human_hall_rate     REAL,             -- n_hallucinated / (n_supported + n_hallucinated)
    llm_hall_rate       REAL,             -- aus note_evals (automatisch)
    agreement_rate      REAL,             -- % identische Claim-Labels (TODO: ersetzen durch Cohen's Kappa)
    PRIMARY KEY (note_path, eval_version) -- Composite PK: mehrere LLM-Versionen vergleichbar
);
"""


def init_db(path: Path = DB_PATH) -> None:
    """Erstellt DB + Schema falls nicht vorhanden. Idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    # Migration für bestehende DBs ohne n_dropped
    try:
        conn.execute("ALTER TABLE pipeline_runs ADD COLUMN n_dropped INT DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE pipeline_runs ADD COLUMN n_words INT DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE pipeline_runs ADD COLUMN model TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE pipeline_runs ADD COLUMN cost_usd REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE note_evals ADD COLUMN anchor_rate REAL")
    except sqlite3.OperationalError:
        pass
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
      n_generated, n_vault, n_inbox, n_merge, n_dropped, n_words, model,
      tokens_total, tokens_input, tokens_output, tokens_cache_read,
      duration_s, eval_version
    """
    data.setdefault("timestamp", datetime.utcnow().isoformat())
    conn.execute("""
        INSERT OR REPLACE INTO pipeline_runs
          (run_id, timestamp, pipeline_version, pdf_source, pdf_key, pdf_label,
           n_generated, n_vault, n_inbox, n_merge, n_dropped, n_words, model,
           cost_usd, tokens_total, tokens_input, tokens_output, tokens_cache_read,
           duration_s, eval_version, fully_cached)
        VALUES
          (:run_id, :timestamp, :pipeline_version, :pdf_source, :pdf_key, :pdf_label,
           :n_generated, :n_vault, :n_inbox, :n_merge, :n_dropped, :n_words, :model,
           :cost_usd, :tokens_total, :tokens_input, :tokens_output, :tokens_cache_read,
           :duration_s, :eval_version, :fully_cached)
    """, {
        "run_id":            data.get("run_id"),
        "timestamp":         data.get("timestamp"),
        "pipeline_version":  data.get("pipeline_version"),
        "pdf_source":        data.get("pdf_source"),
        "pdf_key":           data.get("pdf_key"),
        "pdf_label":         data.get("pdf_label"),
        "n_generated":       data.get("n_generated", 0),
        "n_vault":           data.get("n_vault", 0),
        "n_inbox":           data.get("n_inbox", 0),
        "n_merge":           data.get("n_merge", 0),
        "n_dropped":         data.get("n_dropped", 0),
        "n_words":           data.get("n_words", 0),
        "model":             data.get("model", ""),
        "cost_usd":          data.get("cost_usd", 0.0),
        "tokens_total":      data.get("tokens_total", 0),
        "tokens_input":      data.get("tokens_input", 0),
        "tokens_output":     data.get("tokens_output", 0),
        "tokens_cache_read": data.get("tokens_cache_read", 0),
        "duration_s":        data.get("duration_s", 0.0),
        "eval_version":      data.get("eval_version"),
        "fully_cached":      1 if (data.get("tokens_total", 0) == 0 and data.get("duration_s", 0) > 0) else 0,
    })


def insert_eval(conn: sqlite3.Connection, data: dict) -> None:
    """Schreibt eine Note-Evaluierung in note_evals.

    data-Keys (alle optional ausser eval_id):
      eval_id, run_id, note_path, acceptance_status,
      hallucination_rate, coverage_factual, coverage_rate,
      tokens_total, tokens_input, tokens_output, tokens_cache_read,
      wall_time_s, pipeline_version, pdf, language, eval_version, timestamp
    """
    data.setdefault("timestamp", datetime.utcnow().isoformat())
    conn.execute("""
        INSERT OR REPLACE INTO note_evals
          (eval_id, run_id, note_path, acceptance_status,
           hallucination_rate, coverage_factual, coverage_rate, anchor_rate,
           tokens_total, tokens_input, tokens_output, tokens_cache_read,
           wall_time_s, pipeline_version, pdf, language, eval_version, timestamp)
        VALUES
          (:eval_id, :run_id, :note_path, :acceptance_status,
           :hallucination_rate, :coverage_factual, :coverage_rate, :anchor_rate,
           :tokens_total, :tokens_input, :tokens_output, :tokens_cache_read,
           :wall_time_s, :pipeline_version, :pdf, :language, :eval_version, :timestamp)
    """, {
        "eval_id":           data.get("eval_id"),
        "run_id":            data.get("run_id"),
        "note_path":         data.get("note_path") or data.get("note"),
        "acceptance_status": data.get("acceptance_status"),
        "hallucination_rate":data.get("hallucination_rate"),
        "coverage_factual":  data.get("coverage_factual"),
        "coverage_rate":     data.get("coverage_rate"),
        "anchor_rate":       data.get("anchor_rate"),
        "tokens_total":      data.get("tokens_total"),
        "tokens_input":      data.get("tokens_input"),
        "tokens_output":     data.get("tokens_output"),
        "tokens_cache_read": data.get("tokens_cache_read"),
        "wall_time_s":       data.get("wall_time_s"),
        "pipeline_version":  data.get("pipeline_version") or data.get("version"),
        "pdf":               data.get("pdf"),
        "language":          data.get("language"),
        "eval_version":      data.get("eval_version"),
        "timestamp":         data.get("timestamp"),
    })


# ---------------------------------------------------------------------------
# Read-Helpers fuer Dashboard
# ---------------------------------------------------------------------------

def query_pipeline_runs(path: Path = DB_PATH) -> list[dict]:
    """Alle pipeline_runs als Liste von Dicts."""
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY timestamp"
        ).fetchall()
        return [dict(r) for r in rows]


def query_note_evals(path: Path = DB_PATH,
                     eval_version: str | None = None,
                     pipeline_version: str | None = None) -> list[dict]:
    """note_evals mit optionalen Filtern."""
    where, params = [], []
    if eval_version:
        where.append("eval_version = ?"); params.append(eval_version)
    if pipeline_version:
        where.append("pipeline_version = ?"); params.append(pipeline_version)
    sql = "SELECT * FROM note_evals"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp"
    with get_db(path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def query_kpi_trend(path: Path = DB_PATH,
                    eval_version: str | None = None) -> dict:
    """KPI-Trend pro Pipeline-Version fuer Dashboard-Sparklines."""
    where = "WHERE eval_version = ?" if eval_version else ""
    params = [eval_version] if eval_version else []
    with get_db(path) as conn:
        rows = conn.execute(f"""
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
        """, params).fetchall()
        return [dict(r) for r in rows]


def available_eval_versions(path: Path = DB_PATH) -> list[str]:
    """Alle vorhandenen eval_versions sortiert."""
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT eval_version FROM note_evals "
            "WHERE eval_version IS NOT NULL ORDER BY eval_version"
        ).fetchall()
        return [r[0] for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"DB initialisiert: {DB_PATH}")
