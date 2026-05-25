"""shared/db_schema.py — Kanonisches SQLite-Schema für atomic_analytics.db.

Importiert von generative/db.py und extractive/eval/extractive_eval.py.
Änderungen hier einmalig pflegen — nie in beiden Files separat.
"""

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id            TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    pipeline_version  TEXT,
    pdf_source        TEXT,
    pdf_key           TEXT,
    pdf_label         TEXT,
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
    fully_cached      INT  DEFAULT 0
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

CREATE TABLE IF NOT EXISTS calibration_labels (
    note_path           TEXT NOT NULL,
    eval_version        TEXT NOT NULL DEFAULT '4.1',
    labeled_at          TEXT,
    n_claims            INT  DEFAULT 0,
    n_supported         INT  DEFAULT 0,
    n_hallucinated      INT  DEFAULT 0,
    n_uncertain         INT  DEFAULT 0,
    human_hall_rate     REAL,
    llm_hall_rate       REAL,
    agreement_rate      REAL,
    PRIMARY KEY (note_path, eval_version)
);
"""
