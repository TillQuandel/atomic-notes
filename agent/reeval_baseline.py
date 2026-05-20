"""reeval_baseline.py — Re-Evaluiert alle Baseline-Notes mit eval_quality_v4.

Läuft eval_quality_v4 über alle vault__*.md in .cache/eval/baseline/PDF-Name/
und schreibt Ergebnisse in quality_history.jsonl + atomic_analytics.db.

Resume-fähig: überspringt Notes die bereits mit eval_version=4.1 in DB stehen.

Verwendung:
  python reeval_baseline.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import eval_quality_v4 as _eq
import db as _db

BASELINE_DIR = ROOT / ".cache" / "eval" / "baseline"

# Mapping: Ordner-Präfix → PDF-Pfad
PDF_MAP = {
    "Bates":      Path(r"C:/Users/tillq/OneDrive/Dokumente/Literatur/Bates - 2017 - Information Behavior.pdf"),
    "Hertzum":    Path(r"C:/Users/tillq/OneDrive/Dokumente/Literatur/Hertzum - 2023 - Information seeking by experimentation Trying something out to discover what.pdf"),
    "Kaletski":   Path(r"C:/Users/tillq/OneDrive/Dokumente/Literatur/Kaletski - 2017 - Faculty Perceptions ACRL Framework.pdf"),
    "Kuhlthau":   None,  # PDF nicht verfügbar
    "Schlebbe":   None,  # PDF nicht verfügbar
}


def _find_pdf(folder_name: str) -> Path | None:
    for prefix, pdf_path in PDF_MAP.items():
        if folder_name.startswith(prefix):
            return pdf_path
    return None


def _already_done(note_name: str, conn) -> bool:
    """Prüft ob Note schon mit eval_version=4.1 in DB steht."""
    row = conn.execute(
        "SELECT 1 FROM note_evals WHERE note_path=? AND eval_version='4.1' LIMIT 1",
        (note_name,)
    ).fetchone()
    return row is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BASELINE_DIR.exists():
        print(f"Baseline-Dir nicht gefunden: {BASELINE_DIR}")
        sys.exit(1)

    notes = []
    for pdf_dir in sorted(BASELINE_DIR.iterdir()):
        if not pdf_dir.is_dir():
            continue
        pdf_path = _find_pdf(pdf_dir.name)
        if pdf_path is None:
            print(f"  [skip] kein PDF für: {pdf_dir.name[:50]}")
            continue
        if not pdf_path.exists():
            print(f"  [skip] PDF nicht gefunden: {pdf_path.name}")
            continue
        for note_file in sorted(pdf_dir.glob("vault__*.md")):
            notes.append((note_file, pdf_path, pdf_dir.name))

    print(f"Zu evaluieren: {len(notes)} Notes\n")

    done = skip = errors = 0

    # Reeval-Run in pipeline_runs eintragen damit FK-Constraints erfüllt sind
    from agents.base import _RUN_ID as _reeval_run_id
    if not args.dry_run:
        with _db.get_db() as _conn_init:
            _conn_init.execute("""
                INSERT OR IGNORE INTO pipeline_runs
                (run_id, timestamp, pipeline_version, pdf_source, pdf_key, pdf_label,
                 n_generated, n_vault, n_inbox, fully_cached)
                VALUES (?, datetime('now'), 'reeval', 'baseline-reeval', 'reeval', 'Re-Eval Baseline',
                        ?, 0, 0, 0)
            """, (_reeval_run_id, len(notes)))
        print(f"  Reeval-Run ID: {_reeval_run_id}\n")

    with _db.get_db() as conn:
        for i, (note_path, pdf_path, folder) in enumerate(notes, 1):
            if _already_done(note_path.name, conn):
                print(f"  [{i:2}/{len(notes)}] skip (bereits v4.1): {note_path.name[:55]}")
                skip += 1
                continue

            print(f"  [{i:2}/{len(notes)}] {note_path.name[:55]}")

            if args.dry_run:
                done += 1
                continue

            try:
                result = _eq.eval_note(note_path, pdf_path)
                if "error" not in result:
                    _eq.save_result(result)
                    done += 1
                    hall = result.get("hallucination_rate", -1)
                    cov  = result.get("coverage_factual", -1)
                    print(f"       → hall={hall:.1%}  cov={cov:.1%}")
                else:
                    print(f"       → FEHLER: {result['error']}")
                    errors += 1
            except Exception as e:
                print(f"       → EXCEPTION: {e}")
                errors += 1

    mode = "[dry-run] " if args.dry_run else ""
    print(f"\n{mode}Fertig: {done} neu evaluiert, {skip} übersprungen, {errors} Fehler")

    if not args.dry_run:
        evals = _db.query_note_evals(eval_version="4.1")
        print(f"note_evals mit eval_version=4.1: {len(evals)}")


if __name__ == "__main__":
    main()
