#!/usr/bin/env python3
"""Extrahiert die Anker-Forensik-Tabelle aus `LAUF-INFO.md`-Diagnoseberichten (#232).

Diese Berichte liegen ausserhalb des Repos (OneDrive-Session-Notizen, siehe
`ATOMIC_AGENT_REPORTS_DIR`/CLI-Pfad) und enthalten pro handverifiziertem Lauf
eine Markdown-Tabelle im Format:

    | # | Claim (gekuerzt) | Label | Cosine | Befund |
    |---|---|---|---|---|
    | idx 4 | "..." | not_in_context | 0,614 | **Falsch-negativ.** ... |

Dieses Skript parst NUR diese Tabelle (Schritt 1 der Goldset-Pipeline) und gibt
strukturierte Roh-Zeilen aus -- eine Zwischenstufe, kein fertiges Goldset:

  - `claim_snippet`: der gekuerzte Claim-Text aus dem Bericht (NICHT identisch
    mit dem echten `extract_claims()`-Output der Pipeline -- nur zur
    Wiedererkennung/Zuordnung gedacht).
  - `cosine`: die im Bericht protokollierte Chunk-Top-Cosine (deckt sich bei
    Stichproben exakt mit `RetrievedContext.top_cosine` aus der echten
    Pipeline-Neuberechnung -- siehe Goldset-README).
  - `befund`: der Freitext-Befund/die Adjudikation aus dem Bericht.

Schritt 2 (PDF-Gegenprobe des `evidence_quote` per pdftotext) und Schritt 3
(Zusammenbau von `anchors.jsonl` mit echtem Claim-Text aus `extract_claims()`)
sind bewusst NICHT Teil dieses generischen Parsers -- die PDF-Verifikation
erfordert manuelle/fallspezifische Pruefung (Prinzip "niemals raten") und lebt
in der Provenienz-Dokumentation in `README.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Zeile z.B.: "| idx 4 | \"355 Vergleichsstudien ...\" | not_in_context | 0,614 | **Falsch-negativ.** ... |"
# Die Label-Zelle traegt manchmal Markdown-Bold + Zusatzinfo, z.B.
# "**contradicted** (Audit-Override, urspr. Judge: partially_supported)" -- das
# eigentliche Label wird per _LABEL_TOKEN_RE aus der Zelle herausgeloest.
#
# EDGE-CASE (bewusst nicht abgesichert): enthaelt der Claim- oder Befund-Text
# selbst einen pipe-getrennten `| ... | Zahl |`-foermigen Teilstring (z.B. ein
# eingebettetes Mini-Tabellenfragment oder eine Zahl mit Kontext in Pipes), kann
# die Regex die Spaltengrenzen falsch setzen und still ein falsches Label/Cosine
# ziehen. Das ist akzeptiert, weil dieses Skript nur die ZWISCHENSTUFE (Schritt 1)
# liefert: JEDER so geparste Anker durchlaeuft vor Aufnahme in anchors.jsonl die
# verpflichtende manuelle `pdftotext`-Gegenprobe (Schritt 2, siehe README) --
# ein fehlgeparster Wert wuerde dort auffallen und nie ins Goldset gelangen.
_TABLE_ROW_RE = re.compile(
    r"^\|\s*idx\s*(?P<idx>\d+)\s*\|\s*(?P<claim>.+?)\s*\|\s*(?P<label_cell>[^|]+?)"
    r"\s*\|\s*(?P<cosine>[\d,.]+)\s*\|\s*(?P<befund>.+?)\s*\|\s*$",
    re.MULTILINE,
)
_LABEL_TOKEN_RE = re.compile(r"[a-z_]+")


def parse_anker_forensik_table(md_text: str) -> list[dict]:
    """Parst alle `| idx N | Claim | Label | Cosine | Befund |`-Zeilen einer LAUF-INFO.md.

    Ueberspringt die Header-/Trenner-Zeilen der Tabelle automatisch (die
    Regex verlangt ein numerisches `idx N`, das dort nicht vorkommt).
    """
    rows: list[dict] = []
    for match in _TABLE_ROW_RE.finditer(md_text):
        cosine_raw = match.group("cosine").replace(",", ".")
        try:
            cosine = float(cosine_raw)
        except ValueError:
            continue
        label_cell = match.group("label_cell").strip()
        label_token = _LABEL_TOKEN_RE.search(label_cell.replace("*", ""))
        rows.append(
            {
                "idx": int(match.group("idx")),
                "claim_snippet": match.group("claim").strip().strip('"'),
                "label": label_token.group(0) if label_token else label_cell,
                "label_cell_raw": label_cell,
                "cosine": cosine,
                "befund": match.group("befund").strip(),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Ein oder mehrere LAUF-INFO.md-Dateien (oder Verzeichnisse, die rekursiv nach "
        "LAUF-INFO.md durchsucht werden).",
    )
    parser.add_argument("--out", type=Path, default=None, help="JSONL-Ausgabedatei (Default: stdout).")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("LAUF-INFO.md")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warn: Pfad nicht gefunden, uebersprungen: {p}", file=sys.stderr)

    all_rows: list[dict] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        rows = parse_anker_forensik_table(text)
        for row in rows:
            row["source_report"] = str(f)
        all_rows.extend(rows)
        print(f"{f}: {len(rows)} Anker-Zeile(n) gefunden", file=sys.stderr)

    out_lines = [json.dumps(row, ensure_ascii=False) for row in all_rows]
    if args.out:
        args.out.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        print(f"{len(all_rows)} Zeile(n) -> {args.out}", file=sys.stderr)
    else:
        for line in out_lines:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
