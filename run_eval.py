#!/usr/bin/env python3
"""run_eval.py — Post-Processing LLM-Judge Eval auf generierten Notes.

Nimmt einen extractive- oder generative-Run (run_id + Notes-Verzeichnis + Quell-PDF)
und evaluiert jede Note mit eval_quality_v4.eval_note() (Claude-API).
Ergebnisse landen in note_evals mit eval_version=4.1 und der uebergebenen
pipeline_version — direkt vergleichbar mit generative-Runs im internen Dashboard.

Usage:
    python run_eval.py --run-id <uuid> --notes ./tmp/extractive-bates --pdf bates.pdf
    python run_eval.py --run-id <uuid> --notes ./tmp/extractive-bates --pdf bates.pdf --no-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generative import db as _db
from generative.eval_quality_v4 import eval_note, EVAL_VERSION


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-Processing LLM-Eval (eval_quality_v4) auf generierten Notes")
    ap.add_argument("--run-id", required=True, help="run_id aus pipeline_runs")
    ap.add_argument("--notes", required=True, help="Verzeichnis mit .md-Dateien")
    ap.add_argument("--pdf", required=True, help="Quell-PDF")
    ap.add_argument("--no-cache", action="store_true", help="Claude-Cache umgehen")
    args = ap.parse_args()

    notes_dir = Path(args.notes)
    pdf_path = Path(args.pdf)

    if not notes_dir.is_dir():
        sys.exit(f"Fehler: Notes-Verzeichnis nicht gefunden: {notes_dir}")
    if not pdf_path.exists():
        sys.exit(f"Fehler: PDF nicht gefunden: {pdf_path}")

    runs = _db.query_pipeline_runs()
    run = next((r for r in runs if r["run_id"] == args.run_id), None)
    if not run:
        sys.exit(f"Fehler: run_id '{args.run_id}' nicht in pipeline_runs gefunden")
    pipeline_version = run["pipeline_version"]

    notes = sorted(notes_dir.glob("*.md"))
    if not notes:
        sys.exit(f"Keine .md-Dateien in {notes_dir}")

    print(f"\n=== run_eval.py ===")
    print(f"Run:     {args.run_id}")
    print(f"Version: {pipeline_version}")
    print(f"Notes:   {len(notes)} in {notes_dir}")
    print(f"PDF:     {pdf_path.name}")
    print(f"Eval:    {EVAL_VERSION}\n")

    ok, failed = 0, 0
    for i, note_path in enumerate(notes, 1):
        print(f"[{i}/{len(notes)}] {note_path.name} ...", end=" ", flush=True)
        try:
            result = eval_note(note_path, pdf_path, pipeline_version, no_cache=args.no_cache)
            eval_id = f"{args.run_id}__{note_path.stem}"
            with _db.get_db() as conn:
                _db.insert_eval(
                    conn,
                    {
                        "eval_id": eval_id,
                        "run_id": args.run_id,
                        "note_path": note_path.name,
                        "acceptance_status": None,
                        "hallucination_rate": result.get("hallucination_rate"),
                        "coverage_factual": result.get("coverage_factual"),
                        "coverage_rate": result.get("coverage_rate"),
                        "tokens_total": result.get("tokens_total"),
                        "tokens_input": result.get("tokens_input"),
                        "tokens_output": result.get("tokens_output"),
                        "tokens_cache_read": result.get("tokens_cache_read"),
                        "wall_time_s": result.get("wall_time_s"),
                        "pipeline_version": pipeline_version,
                        "pdf": pdf_path.name,
                        "language": result.get("language"),
                        "eval_version": EVAL_VERSION,
                        "timestamp": result.get("timestamp"),
                    },
                )
            hall = result.get("hallucination_rate")
            cov = result.get("coverage_factual")
            hall_str = f"hall={hall:.0%}" if hall is not None and hall >= 0 else "hall=n/a"
            cov_str = f"cov={cov:.0%}" if cov is not None and cov >= 0 else "cov=n/a"
            print(f"{hall_str} {cov_str}")
            ok += 1
        except Exception as e:
            print(f"FEHLER: {e}")
            failed += 1

    print(f"\n=== Fertig: {ok} evaluiert, {failed} Fehler (eval_version={EVAL_VERSION}) ===")


if __name__ == "__main__":
    main()
