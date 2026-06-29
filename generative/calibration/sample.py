"""Stratifizierte Stichprobenziehung für v4.1-Kalibrierung.

Scannt .cache/eval/baseline/<pdf-folder>/{inbox,vault}__*.md, klassifiziert
pro Note das Sprachpaar (DE→DE, EN→DE, EN→EN) via Pipeline-Detector,
und sampelt deterministisch SAMPLE_PER_PAIR Notes je Sprachpaar.

Output: .cache/eval/calibration/sample.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from generative.eval_quality_v2 import _detect_language_pair, _read_note_body  # noqa: E402
from generative.pipeline.pdf_chunker import pdf_to_text  # noqa: E402

BASELINE = ROOT / ".cache" / "eval" / "baseline"
CALIB = ROOT / ".cache" / "eval" / "calibration"
from generative.config import LITERATURE_DIR as _LIT  # noqa: E402

LITERATUR_DIRS = [_LIT]
SAMPLE_PER_PAIR = 15
TARGET_PAIRS = ("DE→DE", "EN→DE")  # EN→EN aktuell leer (Vault primär deutsch)
SEED = 42
PDF_TEXT_BUDGET = 8000  # chars für Language-Detection


def pdf_language_sample(pdf_text: str, budget: int = PDF_TEXT_BUDGET) -> str:
    """Sampled aus Anfang+Mitte+Ende, um English-Abstract-vor-DE-Body-Bias zu vermeiden."""
    if len(pdf_text) <= budget:
        return pdf_text
    third = budget // 3
    mid_start = max(0, len(pdf_text) // 2 - third // 2)
    return pdf_text[:third] + "\n\n" + pdf_text[mid_start : mid_start + third] + "\n\n" + pdf_text[-third:]


def find_pdf(folder_name: str) -> Path | None:
    stem = folder_name.removesuffix(".pdf")
    candidates = []
    for dir_ in LITERATUR_DIRS:
        if not dir_.exists():
            continue
        for pdf in dir_.glob("*.pdf"):
            if pdf.stem == stem:
                return pdf
            if pdf.stem.replace(" ", "") == stem.replace(" ", ""):
                candidates.append(pdf)
    return candidates[0] if candidates else None


def candidate_notes(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.md") if re.match(r"^(inbox|vault)__", p.name))


def main() -> None:
    if not BASELINE.exists():
        print(f"FEHLER: {BASELINE} existiert nicht", file=sys.stderr)
        sys.exit(1)

    candidates_by_pair: dict[str, list[dict]] = defaultdict(list)
    skipped_no_pdf: list[str] = []
    skipped_other_pair: list[tuple[str, str]] = []

    folders = sorted(p for p in BASELINE.iterdir() if p.is_dir())
    print(f"Scanne {len(folders)} PDF-Folder…")

    for folder in folders:
        pdf = find_pdf(folder.name)
        if pdf is None:
            skipped_no_pdf.append(folder.name)
            continue
        try:
            pdf_snippet = pdf_language_sample(pdf_to_text(pdf, strip_frontmatter=True))
        except Exception as exc:
            print(f"  WARN {folder.name}: PDF-Read-Fehler: {exc}", file=sys.stderr)
            continue

        notes = candidate_notes(folder)
        for note in notes:
            try:
                body = _read_note_body(note)
            except Exception as exc:
                print(f"  WARN {note.name}: Read-Fehler: {exc}", file=sys.stderr)
                continue
            if not body.strip():
                continue
            pair = _detect_language_pair(body, pdf_snippet)
            entry = {
                "note_path": str(note.relative_to(ROOT.parent.parent.parent)).replace("\\", "/"),
                "pdf_path": str(pdf).replace("\\", "/"),
                "folder": folder.name,
                "language_pair": pair,
                "note_name": note.name,
            }
            if pair in TARGET_PAIRS:
                candidates_by_pair[pair].append(entry)
            else:
                skipped_other_pair.append((note.name, pair))

    print("\n=== Candidate-Pool ===")
    for pair in TARGET_PAIRS:
        print(f"  {pair}: {len(candidates_by_pair[pair])} Notes")
    if skipped_other_pair:
        other_counts: dict[str, int] = defaultdict(int)
        for _, p in skipped_other_pair:
            other_counts[p] += 1
        print("  (außerhalb Target):", dict(other_counts))
    if skipped_no_pdf:
        print(
            f"  {len(skipped_no_pdf)} Folder ohne PDF-Match: {skipped_no_pdf[:5]}{'…' if len(skipped_no_pdf) > 5 else ''}"
        )

    rng = random.Random(SEED)
    sample: list[dict] = []
    for pair in TARGET_PAIRS:
        pool = candidates_by_pair[pair]
        if len(pool) < SAMPLE_PER_PAIR:
            print(f"\n  WARN: {pair} hat nur {len(pool)} Notes (Soll: {SAMPLE_PER_PAIR}) — nehme alle")
            sample.extend(pool)
        else:
            picked = rng.sample(pool, SAMPLE_PER_PAIR)
            sample.extend(picked)

    CALIB.mkdir(parents=True, exist_ok=True)
    out = CALIB / "sample.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for entry in sample:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nSample: {len(sample)} Notes → {out}")
    for pair in TARGET_PAIRS:
        count = sum(1 for s in sample if s["language_pair"] == pair)
        print(f"  {pair}: {count}")


if __name__ == "__main__":
    main()
