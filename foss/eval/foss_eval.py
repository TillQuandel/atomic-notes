from __future__ import annotations
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from rapidfuzz import fuzz

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.db_schema import SCHEMA_SQL as _SCHEMA_MIGRATION

_ANCHOR_RE = re.compile(r"\s*\(S\.\s*\d+(?:-\d+)?\)")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA_MIGRATION)
    # Migrations für ältere DBs ohne anchor_rate
    try:
        conn.execute("ALTER TABLE note_evals ADD COLUMN anchor_rate REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _strip(text: str) -> str:
    return _ANCHOR_RE.sub("", text).strip()


def compute_anchor_rate(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if _ANCHOR_RE.search(s)) / len(sentences)


def compute_hallucination_rate(sentences: list[str], fulltext: str, threshold: int = 75) -> float:
    """Anteil Saetze die NICHT im PDF-Volltext auffindbar sind (rapidfuzz).
    Saetze < 12 Zeichen werden uebersprungen (zu kurz fuer sinnvollen Match).
    Anker werden vor dem Vergleich entfernt.
    """
    if not sentences:
        return 0.0
    scorable = [s for s in sentences if len(_strip(s)) >= 12]
    if not scorable:
        return 0.0
    hallucinated = sum(
        1 for s in scorable
        if fuzz.partial_ratio(_strip(s).lower(), fulltext.lower()) < threshold
    )
    return hallucinated / len(scorable)


def insert_foss_run(
    db_path: Path,
    run_id: str,
    pipeline_version: str,
    pdf_source: str,
    n_generated: int,
    n_words: int,
    duration_s: float,
    language: str = "",
) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO pipeline_runs
              (run_id, timestamp, pipeline_version, pdf_source, pdf_key, pdf_label,
               n_generated, n_inbox, model, cost_usd, tokens_total, duration_s,
               eval_version, fully_cached)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0, ?, ?, 0)
        """, (
            run_id,
            datetime.utcnow().isoformat(),
            pipeline_version,
            pdf_source,
            Path(pdf_source).stem.lower().replace(" ", "-"),
            Path(pdf_source).stem,
            n_generated,
            n_generated,  # foss schreibt alles nach output-dir (kein vault-routing)
            "gliner_medium-v2.1",
            duration_s,
            "foss-1.0",
        ))
        conn.commit()
    finally:
        conn.close()


def insert_foss_eval(
    db_path: Path,
    run_id: str,
    note_title: str,
    sentences: list[str],
    fulltext: str,
    source_file: str,
    pipeline_version: str,
    language: str = "",
) -> dict:
    hall_rate = compute_hallucination_rate(sentences, fulltext)
    anch_rate = compute_anchor_rate(sentences)
    result = {
        "eval_id":           str(uuid.uuid4()),
        "run_id":            run_id,
        "note_path":         note_title,
        "acceptance_status": None,
        "hallucination_rate": hall_rate,
        "coverage_factual":  None,
        "coverage_rate":     None,
        "anchor_rate":       anch_rate,
        "tokens_total":      0,
        "tokens_input":      0,
        "tokens_output":     0,
        "tokens_cache_read": 0,
        "wall_time_s":       None,
        "pipeline_version":  pipeline_version,
        "pdf":               source_file,
        "language":          language or None,
        "eval_version":      "foss-1.0",
        "timestamp":         datetime.utcnow().isoformat(),
    }
    conn = _connect(db_path)
    try:
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
        """, result)
        conn.commit()
    finally:
        conn.close()
    return result
