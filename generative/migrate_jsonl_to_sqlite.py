"""migrate_jsonl_to_sqlite.py — Einmalige Migration bestehender JSONL-Daten.

Importiert:
  .cache/quality_history.jsonl  → note_evals
  .cache/runs/*.jsonl           → (Agent-Call-Details, kein pipeline_runs-Eintrag)

Aufruf: python migrate_jsonl_to_sqlite.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db

QUALITY_HISTORY = Path(__file__).parent / ".cache" / "quality_history.jsonl"
RUNS_DIR        = Path(__file__).parent / ".cache" / "runs"


def migrate_quality_history(conn, dry_run: bool = False) -> int:
    if not QUALITY_HISTORY.exists():
        print("quality_history.jsonl nicht gefunden — übersprungen.")
        return 0

    rows = [json.loads(l) for l in QUALITY_HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    migrated = 0
    for r in rows:
        if r.get("error"):
            continue  # Fehler-Rows überspringen

        note_name = r.get("note", "")
        ts        = r.get("timestamp", "")
        pver      = r.get("version") or r.get("pipeline_version") or "unknown"
        eval_id   = f"migrated__{pver}__{note_name}"

        if not dry_run:
            db.insert_eval(conn, {
                "eval_id":           eval_id,
                "run_id":            None,  # kein run_id verfügbar aus alten Daten
                "note_path":         note_name,
                "acceptance_status": None,
                "hallucination_rate":r.get("hallucination_rate"),
                "coverage_factual":  r.get("coverage_factual"),
                "coverage_rate":     r.get("coverage_rate"),
                "tokens_total":      r.get("tokens_total"),
                "tokens_input":      r.get("tokens_input"),
                "tokens_output":     r.get("tokens_output"),
                "tokens_cache_read": r.get("tokens_cache_read"),
                "wall_time_s":       r.get("wall_time_s"),
                "pipeline_version":  pver,
                "pdf":               r.get("pdf"),
                "language":          r.get("language"),
                "eval_version":      r.get("eval_version"),
                "timestamp":         ts,
            })
        migrated += 1

    return migrated


def main() -> None:
    ap = argparse.ArgumentParser(description="Migriert JSONL-Daten nach atomic_analytics.db")
    ap.add_argument("--dry-run", action="store_true", help="Nur zählen, nicht schreiben")
    args = ap.parse_args()

    db.init_db()

    with db.get_db() as conn:
        n_evals = migrate_quality_history(conn, dry_run=args.dry_run)

    mode = "[dry-run]" if args.dry_run else ""
    print(f"{mode} note_evals migriert: {n_evals}")
    print(f"DB: {db.DB_PATH}")

    if not args.dry_run:
        # Ergebnis prüfen
        evals = db.query_note_evals()
        trend = db.query_kpi_trend()
        print(f"Gesamt in DB: {len(evals)} note_evals, {len(trend)} Pipeline-Versionen")
        for row in trend:
            print(f"  {row['pipeline_version']:12}  n={row['n']:3}  "
                  f"hall={row['avg_hall']:.1%}  cov={row['avg_cov']:.1%}")


if __name__ == "__main__":
    main()
