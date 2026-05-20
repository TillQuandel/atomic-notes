"""Läuft eval_quality_v4 auf alle Notes in sample.jsonl.

Resume-fähig: skippt Notes, die bereits in quality_history.jsonl stehen
(via note_path + version-Match). Loggt Fortschritt + ETA.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_quality_v4 import (  # noqa: E402
    _QUALITY_HISTORY as HISTORY_PATH,
    eval_note,
    print_summary,
    save_result,
)
from config import AGENT_VERSION  # noqa: E402

CALIB = ROOT / ".cache" / "eval" / "calibration"
SAMPLE_FILE = CALIB / "sample.jsonl"
from config import VAULT as VAULT_ROOT  # noqa: E402


def already_evaluated(note_path: Path, folder: str, version: str) -> bool:
    """Match via folder/filename, um Namens-Kollisionen über Folder hinweg zu vermeiden."""
    if not HISTORY_PATH.exists():
        return False
    target_name = note_path.name
    target_folder = folder
    with HISTORY_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("version") != version:
                continue
            note_field = entry.get("note", "")
            # Match: filename muss am Ende stehen UND folder im Pfad/pdf-Feld vorkommen
            if not note_field.endswith(target_name):
                continue
            pdf_field = entry.get("pdf", "")
            if target_folder in note_field or target_folder in pdf_field:
                return True
    return False


def main() -> None:
    if not SAMPLE_FILE.exists():
        print(f"FEHLER: {SAMPLE_FILE} fehlt — erst sample.py laufen lassen", file=sys.stderr)
        sys.exit(1)

    samples: list[dict] = []
    with SAMPLE_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Eval v4.1 (AGENT_VERSION={AGENT_VERSION}) auf {len(samples)} Notes")
    print(f"History: {HISTORY_PATH}\n")

    durations: list[float] = []
    failed: list[tuple[str, str]] = []
    skipped = 0

    for idx, entry in enumerate(samples, start=1):
        note_path = VAULT_ROOT / entry["note_path"]
        pdf_path = Path(entry["pdf_path"])
        pair = entry["language_pair"]

        if not note_path.exists():
            print(f"  [{idx}/{len(samples)}] SKIP (note fehlt): {note_path}")
            failed.append((str(note_path), "note-missing"))
            continue
        if not pdf_path.exists():
            print(f"  [{idx}/{len(samples)}] SKIP (pdf fehlt): {pdf_path}")
            failed.append((str(note_path), "pdf-missing"))
            continue
        if already_evaluated(note_path, entry["folder"], AGENT_VERSION):
            print(f"  [{idx}/{len(samples)}] CACHED: {note_path.name}  [{pair}]")
            skipped += 1
            continue

        eta = ""
        if durations:
            avg = sum(durations) / len(durations)
            remaining = (len(samples) - idx + 1) * avg
            eta = f"  (ETA ~{remaining / 60:.1f} min)"
        print(f"  [{idx}/{len(samples)}] RUN [{pair}]: {note_path.name}{eta}")
        t0 = time.time()
        try:
            result = eval_note(note_path, pdf_path, AGENT_VERSION)
            save_result(result)
            dt = time.time() - t0
            durations.append(dt)
            hr = result.get("hallucination_rate", -1)
            cr = result.get("coverage_rate", -1)
            print(f"      → hal={hr:.2f}  cov={cr:.2f}  ({dt:.1f}s)")
        except Exception as exc:
            print(f"      FEHLER: {exc}")
            failed.append((str(note_path), str(exc)[:120]))

    print(f"\n=== Done. Evaluated {len(durations)}, cached {skipped}, failed {len(failed)} ===")
    if failed:
        print("Failures:")
        for path, err in failed[:10]:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
