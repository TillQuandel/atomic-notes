"""eval_batch.py -- Alle vault__*.md in .cache/eval/baseline/ kostenlos evaluieren.

Kein LLM, kein API-Call -- nur PyMuPDF + Fuzzy + Semantic gegen die Quell-PDFs.
Ergebnisse landen in .cache/quality_history.jsonl.

Version-Mapping: Note-Stem → Pipeline-Version wird aus den Log-Dateinamen abgeleitet
(bates_v35_run1.log → v35). So landen Qualitaetsmetriken unter der richtigen
Log-Version und der Dashboard-Versionsfilter funktioniert fuer Fehlerquote/Abdeckung.

Usage: python eval_batch.py [--dry-run]
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


from generative.eval_quality import eval_note, save_result, print_summary
from generative.config import AGENT_VERSION

BASELINE_DIR = Path(__file__).parent / ".cache" / "eval" / "baseline"

# Basis-Pfad zur Literatur — anpassen wenn Vault auf anderem System liegt
_LIT_BASE = Path.home() / "OneDrive" / "Dokumente" / "Literatur"

PDF_MAP = {
    "Bates - 2017 - Information Behavior":
        _LIT_BASE / "Bates - 2017 - Information Behavior.pdf",
    "Kuhlthau - INFORMATION SEARCH PROCESS":
        _LIT_BASE / "Kuhlthau - INFORMATION SEARCH PROCESS.pdf",
    "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis":
        _LIT_BASE / "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf",
    "6.openAI guide to building practical agents":
        _LIT_BASE / "ai-ml-bookshelf" / "6.openAI guide to building practical agents.pdf",
    "Venkatesh et al. - 2003 - User Acceptance of Information Technology Toward A Unified View1":
        _LIT_BASE / "Venkatesh et al. - 2003 - User Acceptance of Information Technology Toward A Unified View1.pdf",
    "Forster - 2017 - Information literacy in the workplace":
        _LIT_BASE / "Forster - 2017 - Information literacy in the workplace.pdf",
}

# Patterns
_NOTE_RE = re.compile(r"^\s*\[DRY-RUN\] -> (?:Vault|Inbox)[^:]*: (.+?)\.md\b")
_VER_RE  = re.compile(r"_(v[\d.]+(?:\.\d+)*)(?:_run\d+)?\.log$")
_KEY_RE  = re.compile(r"^([a-z]+)_")


def _build_note_version_map() -> dict[str, str]:
    """Baut {folder/note_stem: version} aus allen Log-Dateien.

    Ordner-Name kommt direkt aus BASELINE_DIR-Unterverzeichnissen (= PDF-Stem).
    Reverse-Mapping: kurzer Log-Key (bates/kuhlthau/...) → Ordner-Name,
    automatisch aus vorhandenen Unterverzeichnissen abgeleitet — kein _KEY_TO_FOLDER nötig.
    """
    # Automatisches Reverse-Mapping: kurzkey → echter Ordner-Name
    key_to_folder: dict[str, str] = {}
    for folder_dir in BASELINE_DIR.iterdir():
        if not folder_dir.is_dir():
            continue
        short = folder_dir.name.split()[0].lower().rstrip("-")
        key_to_folder[short] = folder_dir.name

    note_version: dict[str, str] = {}

    def ver_key(v: str) -> tuple:
        return tuple(int(n) for n in re.findall(r"\d+", v))

    for log in sorted(BASELINE_DIR.glob("*.log")):
        pdf_key_m = _KEY_RE.match(log.stem)
        if not pdf_key_m:
            continue
        pdf_key = pdf_key_m.group(1)
        ver_m   = _VER_RE.search(log.name)
        ver     = ver_m.group(1) if ver_m else "unknown"
        folder  = key_to_folder.get(pdf_key)
        if not folder:
            continue

        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _NOTE_RE.match(line)
            if not m:
                continue
            stem     = m.group(1)
            full_key = f"{folder}/{stem}"
            existing = note_version.get(full_key)
            if existing is None or ver_key(ver) > ver_key(existing):
                note_version[full_key] = ver

    print(f"[eval_batch] Versions-Mapping: {len(note_version)} Notes -> Version zugeordnet")
    return note_version


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Nur ausgeben, nicht speichern")
    args = ap.parse_args()

    note_version_map = _build_note_version_map()

    notes = list(BASELINE_DIR.rglob("vault__*.md"))
    print(f"[eval_batch] {len(notes)} Notes gefunden")

    ok = skipped = errors = 0

    for note_path in sorted(notes):
        folder = note_path.parent.name
        pdf_path = PDF_MAP.get(folder)
        if pdf_path is None:
            print(f"  [skip] kein PDF-Mapping fuer: {folder}")
            skipped += 1
            continue
        if not pdf_path.exists():
            print(f"  [skip] PDF nicht gefunden: {pdf_path.name}")
            skipped += 1
            continue

        # Version aus Log-Mapping; Fallback auf AGENT_VERSION
        note_stem = note_path.stem
        full_key  = f"{folder}/{note_stem}"
        version   = note_version_map.get(full_key, AGENT_VERSION)

        try:
            result = eval_note(note_path, pdf_path, pipeline_version=version)
            print_summary(result)
            if not args.dry_run and "error" not in result:
                save_result(result)
                ok += 1
            elif "error" in result:
                errors += 1
        except Exception as e:
            print(f"  [ERROR] {note_path.name}: {e}")
            errors += 1

    print(f"\n[eval_batch] Fertig: {ok} gespeichert, {skipped} uebersprungen, {errors} Fehler")


if __name__ == "__main__":
    main()
