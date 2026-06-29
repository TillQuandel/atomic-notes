"""Parsed die vom User gefüllten Label-Tabellen aus calibration/notes/*.md
→ labels_human.jsonl (eine Zeile pro gelabeltem Claim).

MD-Tabellen-Format pro Claim:
    | Label (s/h/?) | Tag (optional) | Notiz |
    |---|---|---|
    | <!--claim_idx=0--> s | tag_value | notiz_text |

Der `<!--claim_idx=N-->`-Kommentar markiert das Claim. User schreibt
Label in dieselbe Zelle (vor oder nach dem Kommentar).

Validiert: label ∈ {s, h, ?}, tag ∈ {evident_*, subtle_*, ""}.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / ".cache" / "eval" / "calibration"
NOTES_DIR = ROOT / "calibration" / "labels-active"
SAMPLE_FILE = CALIB / "sample.jsonl"
OUTPUT = CALIB / "labels_human.jsonl"

VALID_LABELS = {"s", "h", "?"}
VALID_TAGS = {"evident_conflict", "evident_baseless", "subtle_conflict", "subtle_baseless", ""}

ROW_RE = re.compile(
    r"\|\s*<!--claim_idx=(?P<idx>\d+)-->\s*(?P<label>[^|]*)\|"
    r"\s*(?P<tag>[^|]*)\|\s*(?P<notiz>[^|]*)\|",
    re.MULTILINE,
)
FILENAME_RE = re.compile(r"^(?P<idx>\d{2})__")


def load_sample_lookup() -> dict[int, dict]:
    """sample-Index (1-basiert) → sample-entry."""
    lookup: dict[int, dict] = {}
    with SAMPLE_FILE.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if line:
                lookup[i] = json.loads(line)
    return lookup


LABEL_CELL_RE = re.compile(r"^\s*(?P<label>[shSH?])\s*$")


def parse_label_cell(raw: str) -> str | None:
    """Strikt: Zelle muss exakt s, h oder ? enthalten (case-insensitive)."""
    match = LABEL_CELL_RE.match(raw)
    if not match:
        return None
    return match.group("label").lower()


def parse_md(md_text: str) -> list[dict]:
    rows: list[dict] = []
    for match in ROW_RE.finditer(md_text):
        idx = int(match.group("idx"))
        label = parse_label_cell(match.group("label"))
        tag_raw = match.group("tag").strip()
        notiz_raw = match.group("notiz").strip()
        rows.append(
            {
                "claim_idx": idx,
                "label": label,
                "tag": tag_raw if tag_raw in VALID_TAGS else "",
                "tag_invalid": tag_raw if tag_raw and tag_raw not in VALID_TAGS else None,
                "notiz": notiz_raw,
            }
        )
    return rows


def note_claim_agreement(
    human_claims: dict[int, str], pipeline_labels: dict[tuple[str, int], str], note: str
) -> float | None:
    """Claim-level Übereinstimmung Mensch↔Pipeline für EINE Note (#24).

    Paart Human- und Pipeline-Label pro `claim_idx` und liefert den Anteil
    übereinstimmender Claims, oder None wenn es keine gemeinsam gelabelten Claims
    gibt. Ersetzt die alte Aggregat-Differenz `1 - |human_hall - llm_hall|`, die
    zwei Notes mit gleicher Halluzinationsrate als 100% einig zeigte, auch wenn
    sie bei keinem einzelnen Claim übereinstimmten.
    """
    pairs = [
        (label, pipeline_labels[(note, idx)]) for idx, label in human_claims.items() if (note, idx) in pipeline_labels
    ]
    if not pairs:
        return None
    agree = sum(1 for h, p in pairs if h == p)
    return round(agree / len(pairs), 4)


def main() -> None:
    if not NOTES_DIR.exists():
        print(f"FEHLER: {NOTES_DIR} fehlt — erst build_labels.py laufen", file=sys.stderr)
        sys.exit(1)
    sample = load_sample_lookup()

    total_claims = 0
    labeled = 0
    empty = 0
    invalid: list[str] = []
    collected: list[dict] = []
    missing_per_note: dict[str, list[int]] = {}
    invalid_per_note: dict[str, list[int]] = {}

    md_files = sorted(NOTES_DIR.glob("*.md"))
    md_files = [p for p in md_files if not p.name.startswith("INDEX")]
    print(f"Parse {len(md_files)} Label-Files…\n")

    for md in md_files:
        m = FILENAME_RE.match(md.name)
        if not m:
            continue
        sample_idx = int(m.group("idx"))
        sample_entry = sample.get(sample_idx)
        if sample_entry is None:
            print(f"  WARN {md.name}: kein Sample-Eintrag #{sample_idx}", file=sys.stderr)
            continue

        rows = parse_md(md.read_text(encoding="utf-8"))
        for row in rows:
            total_claims += 1
            if row["label"] is None:
                empty += 1
                missing_per_note.setdefault(md.name, []).append(row["claim_idx"])
                continue
            if row["label"] not in VALID_LABELS:
                invalid.append(f"{md.name} claim {row['claim_idx']}: '{row['label']}'")
                invalid_per_note.setdefault(md.name, []).append(row["claim_idx"])
                continue
            if row["tag_invalid"]:
                print(
                    f"  WARN {md.name} claim {row['claim_idx']}: tag '{row['tag_invalid']}' ungültig — gespeichert als ''",
                    file=sys.stderr,
                )
            labeled += 1
            collected.append(
                {
                    "note": sample_entry["note_name"],
                    "language_pair": sample_entry["language_pair"],
                    "claim_idx": row["claim_idx"],
                    "label": row["label"],
                    "tag": row["tag"],
                    "notiz": row["notiz"],
                    "sample_idx": sample_idx,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in collected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # DB: calibration_labels pro Note aggregieren
    try:
        import sys as _sys
        from generative import db as _db
        import sqlite3 as _sq
        from datetime import datetime as _dt

        # Aggregiere Labels pro Note + sammle claim-level Human-Labels (#24)
        from collections import defaultdict

        per_note: dict = defaultdict(lambda: {"s": 0, "h": 0, "?": 0})
        human_by_note: dict[str, dict[int, str]] = defaultdict(dict)
        for row in collected:
            per_note[row["note"]][row["label"]] += 1
            prev = human_by_note[row["note"]].get(row["claim_idx"])
            if prev is not None and prev != row["label"]:
                print(
                    f"  [warn] widersprüchliche Labels für {row['note']} claim "
                    f"{row['claim_idx']}: '{prev}' vs. '{row['label']}' — letztes gewinnt, "
                    f"Counts zählen beide",
                    file=_sys.stderr,
                )
            human_by_note[row["note"]][row["claim_idx"]] = row["label"]

        # Pipeline-Label pro (note, claim_idx) für claim-level Agreement (SSoT: kappa.py).
        # Bei fehlender quality_history bleibt agreement_rate None statt zu crashen —
        # aber laut (sonst sieht ein Versions-Bump wie 0 % Agreement aus). Bewusst KEIN
        # stiller Fallback auf andere Versionen (Provenance: Labels müssen zum Run passen).
        pipeline_labels: dict[tuple[str, int], str] = {}
        if per_note:
            try:
                from generative.calibration.kappa import extract_pipeline_labels, load_pipeline_results
                from generative.config import AGENT_VERSION

                pipeline_labels = extract_pipeline_labels(load_pipeline_results())
                if not pipeline_labels:
                    print(
                        f"  [warn] 0 Pipeline-Labels für AGENT_VERSION={AGENT_VERSION} in "
                        f"quality_history.jsonl — agreement_rate bleibt None (Labels evtl. "
                        f"unter älterer Version erzeugt)",
                        file=_sys.stderr,
                    )
            except (ImportError, OSError, KeyError) as _ke:
                print(f"  [warn] Pipeline-Labels für Agreement nicht ladbar: {_ke}", file=_sys.stderr)

        # LLM-Halluzinationsraten aus note_evals holen (kanonische DB: db.DB_PATH —
        # nicht generative/.cache, dort liegt keine DB)
        _db.init_db(_db.DB_PATH)
        conn_plain = _sq.connect(str(_db.DB_PATH))
        llm_rates = {
            r[0]: r[1]
            for r in conn_plain.execute(
                "SELECT note_path, hallucination_rate FROM note_evals WHERE eval_version='4.1'"
            ).fetchall()
        }
        conn_plain.close()

        with _db.get_db(_db.DB_PATH) as conn:
            for note_name, counts in per_note.items():
                n_s, n_h, n_q = counts["s"], counts["h"], counts["?"]
                n_valid = n_s + n_h
                human_hall = round(n_h / n_valid, 4) if n_valid > 0 else None
                llm_hall = llm_rates.get(note_name)
                # Claim-level statt Aggregat-Differenz (#24)
                agree = note_claim_agreement(human_by_note.get(note_name, {}), pipeline_labels, note_name)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO calibration_labels
                    (note_path, eval_version, labeled_at, n_claims, n_supported, n_hallucinated,
                     n_uncertain, human_hall_rate, llm_hall_rate, agreement_rate)
                    VALUES (?, '4.1', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (note_name, _dt.utcnow().isoformat(), n_s + n_h + n_q, n_s, n_h, n_q, human_hall, llm_hall, agree),
                )
        print(f"  DB: {len(per_note)} Notes in calibration_labels eingetragen")
    except Exception as _e:
        print(f"  [warn] DB-Write fehlgeschlagen: {_e}", file=_sys.stderr)

    print(f"=== Zusammenfassung ===")
    print(f"  Claims total:    {total_claims}")
    print(f"  Gelabelt:        {labeled}")
    print(f"  Leer:            {empty}")
    print(f"  Invalid Labels:  {len(invalid)}")
    print(f"\nOutput: {OUTPUT}")

    if missing_per_note or invalid_per_note:
        print("\n!!! UNVOLLSTÄNDIG !!!")
        if missing_per_note:
            print("  Leere Label-Zellen (pro Datei → claim_idx-Liste):")
            for name, idxs in missing_per_note.items():
                print(f"    - {name}: {idxs}")
        if invalid_per_note:
            print("  Invalide Label-Zellen (nicht s/h/?):")
            for name, idxs in invalid_per_note.items():
                print(f"    - {name}: {idxs}")
        print("\n  → Selection-Bias-Risiko: kappa.py wird sich weigern bei Unvollständigkeit.")
        print("  → Fülle alle Zellen, dann re-run collect.py.")
        sys.exit(2)


if __name__ == "__main__":
    main()
