"""Gold-Set-Generator für das Faithfulness-Gate (E5b, #69).

Baut aus gerenderten Pipeline-Notes (Inbox/atomic-review) Label-Dateien im
Stil von `generative/calibration/labels-active/` — pro Note eine Markdown-
Datei mit den High-Risk-Claims (via `decompose_claims`), einem Kontext-
Snippet der Anker-Seite und einer Label-Tabelle für den Human-Pass (#123).
Zusätzlich eine maschinenlesbare `claims.jsonl` als Grundlage für den
Kalibrierungslauf (Gate über das Gold-Set).

Zwei Normalisierungen, ohne die gerenderte Notes nicht gate-fähig sind:

1. **Footnote→Seite-Mapping**: gerenderte Notes tragen Seitenanker in
   `[^i]`-Definitionen („Label, S. N."). `decompose_claims` erwartet Inline-
   Anker `(S. N)`. Inline-Marker werden ersetzt, Definitionszeilen bleiben
   unangetastet.
2. **Anker-Offset-Mapping**: Alt-Notes (vor PR #79) zählen Form-Feed-Seiten
   ab 1; der heutige `build_page_index` liefert PageLabel-Keys (z. B. 51–55
   beim Hrastinski-PDF). Der Offset wird pro PDF aus den Index-Keys
   abgeleitet — nie geraten: passt beides nicht zusammen, bricht das Tool ab.

Kein ML-Load: das Kontext-Snippet wird lexikalisch gewählt (Token-Überlapp),
nicht per Embedding — das Tool muss ohne Modell-Cache laufen können.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from generative.pipeline.claims import decompose_claims
from generative.pipeline.page_index import build_page_index
from generative.pipeline.pdf_chunker import pdf_to_text

# `[^i]: <Label>, S. N` — Definitionszeile mit Seitenangabe.
_FOOTNOTE_DEF_PAGE_RE = re.compile(r"^\[\^([^\]]+)\]:\s*.*?S\.\s*(\d+)", re.MULTILINE)
# Inline-Marker `[^i]` — negatives Lookahead schließt Definitionszeilen aus.
_INLINE_MARKER_RE = re.compile(r"\[\^([^\]]+)\](?!:)")
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_WORD_RE = re.compile(r"[\wäöüß]{4,}", re.IGNORECASE)


def footnote_page_map(body: str) -> dict[str, int]:
    """Mappt Footnote-Label → Seitenzahl aus den `[^i]:`-Definitionen."""
    return {m.group(1): int(m.group(2)) for m in _FOOTNOTE_DEF_PAGE_RE.finditer(body)}


def inline_page_anchors(body: str, page_map: dict[str, int], offset: int) -> str:
    """Ersetzt Inline-`[^i]`-Marker durch ` (S. N+offset)`; Defs bleiben."""

    def _repl(match: re.Match[str]) -> str:
        label = match.group(1)
        if label in page_map:
            return f" (S. {page_map[label] + offset})"
        return match.group(0)

    return _INLINE_MARKER_RE.sub(_repl, body)


def detect_page_offset(note_pages: set[int], index_keys: list[int]) -> int:
    """Leitet den Anker-Offset aus Note-Seiten und page_index-Keys ab.

    - Note-Seiten ⊆ Index-Keys → 0 (aktuelle Notes, PageLabel-Anker)
    - alle Note-Seiten < kleinster Key → `min(keys) - 1` (Alt-Notes zählen
      Form-Feed ab 1; Form-Feed-Seite 1 entspricht dem kleinsten Key)
    - sonst → ValueError (kein Raten — Prinzip „niemals raten")
    """
    if not note_pages:
        return 0
    keys = set(index_keys)
    if note_pages <= keys:
        return 0
    if max(note_pages) < min(keys):
        offset = min(keys) - 1
        if {p + offset for p in note_pages} <= keys:
            return offset
    raise ValueError(f"Anker-Offset nicht ableitbar: Note-Seiten {sorted(note_pages)} vs. Index-Keys {sorted(keys)}")


def best_snippet(claim_text: str, page_text: str, width: int = 900) -> str:
    """Lexikalisch bester Ausschnitt der Seite: Fenster mit maximalem
    Token-Überlapp (Wörter ≥4 Zeichen) zum Claim, mit Ellipsen getrimmt."""
    if len(page_text) <= width:
        return page_text
    claim_tokens = {t.lower() for t in _WORD_RE.findall(claim_text)}
    best_start, best_score = 0, -1
    step = max(1, width // 3)
    for start in range(0, len(page_text) - width + 1, step):
        chunk = page_text[start : start + width]
        score = sum(1 for t in _WORD_RE.findall(chunk) if t.lower() in claim_tokens)
        if score > best_score:
            best_start, best_score = start, score
    snippet = page_text[best_start : best_start + width].strip()
    prefix = "…" if best_start > 0 else ""
    suffix = "…" if best_start + width < len(page_text) else ""
    return f"{prefix}{snippet}{suffix}"


def _slug(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-äöüÄÖÜß]+", "-", stem).strip("-")
    return stem


def build_goldset(
    notes: list[tuple[str, str]],
    page_index: dict[int, str],
    *,
    source_label: str,
    start_index: int = 1,
) -> tuple[dict[str, str], list[dict]]:
    """Pure Kernfunktion: Notes + page_index → Label-Dateien + Claim-Records.

    `notes` = Liste (Dateiname, Roh-Markdown mit Frontmatter). Rückgabe:
    (Dateiname → Markdown-Inhalt, claims.jsonl-Records).
    """
    files: dict[str, str] = {}
    all_claims: list[dict] = []
    keys = sorted(page_index)

    for i, (note_name, raw) in enumerate(notes):
        body = _FRONTMATTER_RE.sub("", raw)
        page_map = footnote_page_map(body)
        note_pages = set(page_map.values())
        offset = detect_page_offset(note_pages, keys)
        prepared = inline_page_anchors(body, page_map, offset)

        claims = [c for c in decompose_claims(prepared) if not c.is_quote]

        lines = [
            f"# Faithfulness-Label {start_index + i:02d} — {note_name}",
            "",
            f"- **Quelle**: {source_label}",
            f"- **Anker-Offset**: +{offset}",
            f"- **Claims total**: {len(claims)}",
            "",
            "## Härtung",
            "- Pre-Label gegen das PDF verifizieren, nicht gegen das Snippet allein.",
            "- Labels: `s` = supported, `m` = misattributed, `e` = extrapolated, `?` = genuin unklar.",
            "",
            "## Claims",
            "",
        ]

        for idx, claim in enumerate(claims):
            lines.append(f"### Claim {idx + 1}")
            lines.append("")
            lines.append(f"> {claim.text}")
            lines.append("")
            lines.append(f"**Anker-Seite**: {claim.anchor_page if claim.anchor_page is not None else '_keine_'}")
            lines.append(f"**Risk-Types**: {', '.join(claim.risk_types)}")
            lines.append("")
            if claim.anchor_page is not None and claim.anchor_page in page_index:
                snippet = best_snippet(claim.text, page_index[claim.anchor_page])
                lines.append(f"**Kontext (S. {claim.anchor_page}, lexikalisch getrimmt):**")
                lines.append("")
                lines.append(f"- {snippet}")
                lines.append("")
            lines.append("| Pre-Label (s/m/e/?) | Human-Label | Notiz |")
            lines.append("|---|---|---|")
            lines.append(f"| <!--claim_idx={idx}--> | | |")
            lines.append("")

            all_claims.append(
                {
                    "note": note_name,
                    "claim_idx": idx,
                    "text": claim.text,
                    "anchor_page": claim.anchor_page,
                    "risk_types": claim.risk_types,
                    "source": source_label,
                }
            )

        fname = f"{start_index + i:02d}__{_slug(note_name)}.md"
        files[fname] = "\n".join(lines)

    return files, all_claims


def _run_gate(body: str, page_index: dict[int, str], citation):
    """Dünner Indirektions-Punkt: lazy Import, damit das Tool ohne ML-Deps
    läuft, solange `--gate` nicht gesetzt ist — und Tests mocken können."""
    from generative.pipeline.faithfulness_gate import run_faithfulness_gate

    return run_faithfulness_gate(body, page_index, citation)


def gate_verdicts(notes: list[tuple[str, str]], page_index: dict[int, str], *, author: str | None) -> list[dict]:
    """Gate-Lauf über die vorbereiteten Notes → ein Record je Claim-Verdikt.

    Nutzt dieselbe Vorbereitung (Footnote→Inline, Offset) wie `build_goldset`;
    `decompose_claims` ist deterministisch, daher entspricht die Verdikt-
    Reihenfolge der `claim_idx`-Reihenfolge der Label-Dateien.
    """
    from generative.schemas.citation import CitationMeta

    keys = sorted(page_index)
    records: list[dict] = []
    for note_name, raw in notes:
        body = _FRONTMATTER_RE.sub("", raw)
        page_map = footnote_page_map(body)
        offset = detect_page_offset(set(page_map.values()), keys)
        prepared = inline_page_anchors(body, page_map, offset)

        citation = CitationMeta(author=author, year=None, title=None, doi=None, source_file=note_name)
        result = _run_gate(prepared, page_index, citation)
        for idx, verdict in enumerate(result.verdicts):
            records.append(
                {
                    "note": note_name,
                    "claim_idx": idx,
                    "status": verdict.status,
                    "entailment": verdict.entailment,
                    "evidence": verdict.evidence,
                }
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gold-Set-Label-Dateien fürs Faithfulness-Gate erzeugen")
    parser.add_argument("pdf", type=Path, help="Quell-PDF (für page_index)")
    parser.add_argument("--notes", type=Path, nargs="+", required=True, help="Note-Dateien oder -Verzeichnisse")
    parser.add_argument("--out", type=Path, required=True, help="Ausgabe-Verzeichnis")
    parser.add_argument("--source-label", required=True, help='Quellen-Label, z. B. "Hrastinski 2008"')
    parser.add_argument("--start-index", type=int, default=1, help="Laufende Nummer der ersten Label-Datei")
    parser.add_argument("--gate", action="store_true", help="zusätzlich Gate-Lauf → verdicts.jsonl (lädt ML-Modelle)")
    parser.add_argument("--author", default=None, help="Primärautor für den Gate-Lauf (CitationMeta)")
    args = parser.parse_args(argv)

    note_paths: list[Path] = []
    for entry in args.notes:
        if entry.is_dir():
            note_paths.extend(sorted(entry.glob("*.md")))
        else:
            note_paths.append(entry)

    page_index = build_page_index(pdf_to_text(args.pdf))
    if not page_index:
        print(f"FEHLER: kein [S. N]-Index aus {args.pdf}")
        return 1

    notes = [(p.name, p.read_text(encoding="utf-8")) for p in note_paths]
    files, claims = build_goldset(notes, page_index, source_label=args.source_label, start_index=args.start_index)

    args.out.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (args.out / fname).write_text(content, encoding="utf-8")
    jsonl_path = args.out / "claims.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as fh:
        for record in claims:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(files)} Label-Dateien, {len(claims)} Claims → {args.out}")

    if args.gate:
        verdicts = gate_verdicts(notes, page_index, author=args.author)
        verdicts_path = args.out / "verdicts.jsonl"
        with verdicts_path.open("a", encoding="utf-8") as fh:
            for record in verdicts:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{len(verdicts)} Gate-Verdicts → {verdicts_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
