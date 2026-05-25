# extractive/orchestrator.py
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractive.eval.extractive_eval import (
    compute_anchor_rate,
    compute_hallucination_rate,
    insert_extractive_run,
)
from extractive.pipeline.adapter import write_note
from extractive.pipeline.gliner_planner import plan_concepts
from extractive.pipeline.pdf_chunker import detect_language, extract_chunks, extract_fulltext
from extractive.pipeline.sentence_extractor import add_page_anchors, extract_body_for_concept
from shared.schemas.atomic_note_extractive import AtomicNoteExtractive

EXTRACTIVE_VERSION = "extractive-v0.2.0"


def main():
    ap = argparse.ArgumentParser(description="extractive: PDF -> Atomic Notes (kein API)")
    ap.add_argument("--source", required=True, help="Pfad zur PDF-Datei")
    ap.add_argument("--output", default="obsidian", choices=["obsidian", "md", "json"])
    ap.add_argument("--out-dir", default="./output", help="Ausgabe-Verzeichnis")
    ap.add_argument("--eval-db", default=None, help="Pfad zur atomic_analytics.db (z.B. ../generative/.cache/atomic_analytics.db)")
    ap.add_argument("--eval-jsonl", default=None, help="Optionaler JSONL-Eval-Output fuer Tests/Benchmarks")
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

    print(f"\n=== extractive {EXTRACTIVE_VERSION} ===\n")

    print("[1] PDF extrahieren...")
    chunks = extract_chunks(source)
    fulltext = extract_fulltext(source)
    word_count = sum(len(c.text.split()) for c in chunks)
    print(f"    {len(chunks)} Chunks, {word_count} Woerter")

    lang = detect_language(fulltext[:500])
    if lang != "en":
        print(f"    WARNUNG: Sprache erkannt = '{lang}' (v1 = Englisch only)")

    print("[2] Konzepte extrahieren (GLiNER)...")
    concepts = plan_concepts(chunks, main_language=lang)
    print(f"    {len(concepts)} Konzepte")

    print("[3] Saetze extrahieren (LexRank)...")
    notes = []
    for c in concepts:
        body = add_page_anchors(
            extract_body_for_concept(c["name"], fulltext),
            [c.get("page", 1)],
        )
        if not body:
            continue
        note = AtomicNoteExtractive(
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

    if args.eval_jsonl:
        eval_path = Path(args.eval_jsonl)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with eval_path.open("w", encoding="utf-8") as f:
            for note in notes:
                row = {
                    "title": note.title,
                    "hallucination_rate": compute_hallucination_rate(note.extracted_body, fulltext),
                    "anchor_rate": compute_anchor_rate(note.extracted_body),
                    "pipeline_version": EXTRACTIVE_VERSION,
                    "source_file": source.name,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[5] Eval JSONL -> {eval_path}")

    if args.eval_db:
        print("[5] Run in DB registrieren...")
        db_path = Path(args.eval_db)
        duration_s = time.monotonic() - t_start
        insert_extractive_run(
            db_path=db_path,
            run_id=run_id,
            pipeline_version=EXTRACTIVE_VERSION,
            pdf_source=source.name,
            n_generated=len(notes),
            n_words=word_count,
            duration_s=duration_s,
            language=lang,
        )
        print(f"    run_id={run_id}")
        print(f"    Eval: python run_eval.py --run-id {run_id} --notes <out-dir> --pdf <pdf>")

    total = sum(len(n.extracted_body) for n in notes)
    print(f"\n=== Fertig: {len(notes)} Notes, {total} Saetze ===")


if __name__ == "__main__":
    main()
