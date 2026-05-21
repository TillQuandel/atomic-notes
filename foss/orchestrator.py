# foss/orchestrator.py
from __future__ import annotations
import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foss.pipeline.pdf_chunker import extract_chunks, extract_fulltext, detect_language
from foss.pipeline.gliner_planner import plan_concepts
from foss.pipeline.sentence_extractor import extract_body_for_concept, add_page_anchors
from foss.pipeline.adapter import write_note
from foss.eval.foss_eval import insert_foss_run, insert_foss_eval
from shared.schemas.atomic_note_foss import AtomicNoteFoss

FOSS_VERSION = "foss-v0.1.1"


def main():
    ap = argparse.ArgumentParser(description="foss-atomic: PDF -> Atomic Notes (kein API)")
    ap.add_argument("--source", required=True, help="Pfad zur PDF-Datei")
    ap.add_argument("--output", default="obsidian", choices=["obsidian", "md", "json"])
    ap.add_argument("--out-dir", default="./output", help="Ausgabe-Verzeichnis")
    ap.add_argument("--eval-db", default=None, help="Pfad zur atomic_analytics.db (z.B. ../agent/.cache/atomic_analytics.db)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true", help="Keine Dateien schreiben, nur Eval")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        sys.exit(f"Fehler: PDF nicht gefunden: {source}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    t_start = time.monotonic()

    print(f"\n=== foss-atomic {FOSS_VERSION} ===\n")

    print("[1] PDF extrahieren...")
    chunks = extract_chunks(source)
    fulltext = extract_fulltext(source)
    word_count = sum(len(c.text.split()) for c in chunks)
    print(f"    {len(chunks)} Chunks, {word_count} Woerter")

    lang = detect_language(fulltext[:500])
    if lang != "en":
        print(f"    WARNUNG: Sprache erkannt = '{lang}' (v1 = Englisch only)")

    print("[2] Konzepte extrahieren (GLiNER)...")
    concepts = plan_concepts(chunks)
    print(f"    {len(concepts)} Konzepte")

    print("[3] Saetze extrahieren (LexRank)...")
    notes = []
    for c in concepts:
        body = add_page_anchors(
            extract_body_for_concept(c["name"], fulltext),
            [c.get("page", 1)]
        )
        note = AtomicNoteFoss(
            title=c["name"],
            concept_type=c["type"],
            extracted_body=body,
            source_anchors=[{"page": c.get("page", 1), "quote": body[0][:80] if body else "", "score": 1.0}],
            source_file=source.name,
            created=datetime.now().strftime("%Y-%m-%d"),
        )
        notes.append(note)

    print(f"[4] {len(notes)} Notes {'(dry-run, kein Schreiben)' if args.dry_run else f'-> {out_dir}/'}...")
    if not args.dry_run:
        for note in notes:
            p = write_note(note, out_dir, args.output)
            print(f"    -> {p.name}")

    if args.eval_db:
        print("[5] Eval -> DB...")
        db_path = Path(args.eval_db)
        duration_s = time.monotonic() - t_start
        insert_foss_run(
            db_path=db_path,
            run_id=run_id,
            pipeline_version=FOSS_VERSION,
            pdf_source=source.name,
            n_generated=len(notes),
            n_words=word_count,
            duration_s=duration_s,
            language=lang,
        )
        for note in notes:
            insert_foss_eval(
                db_path=db_path,
                run_id=run_id,
                note_title=note.title,
                sentences=note.extracted_body,
                fulltext=fulltext,
                source_file=source.name,
                pipeline_version=FOSS_VERSION,
                language=lang,
            )
        print(f"    {len(notes)} Eval-Eintraege in {db_path}")

    total = sum(len(n.extracted_body) for n in notes)
    print(f"\n=== Fertig: {len(notes)} Notes, {total} Saetze ===")


if __name__ == "__main__":
    main()
