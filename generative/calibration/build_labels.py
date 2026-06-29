"""Erzeugt Labeling-Markdowns für die 30 Sample-Notes.

Pro Note ein .md mit Claims-Tabelle + Top-3-Kontext-Snippets (±500 chars).
**Pipeline-Verdict wird NICHT angezeigt** (Härtung #1).

User editiert die Label-Spalte im MD direkt. Späteres collect.py parsed die
Tabellen → labels_human.jsonl.

Output: .cache/eval/calibration/notes/<NN>__<slug>.md + INDEX.md
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from generative.config import AGENT_VERSION, QUALITY_HISTORY as HISTORY_PATH  # noqa: E402

CALIB = ROOT / ".cache" / "eval" / "calibration"
SAMPLE_FILE = CALIB / "sample.jsonl"
NOTES_DIR = ROOT / "calibration" / "labels-active"
BLIND_DIR = ROOT / "calibration" / "labels-active-blind"
CONTEXT_RADIUS = 500  # chars vor/nach Match-Mitte
TOP_K_CONTEXTS = 3
BLIND_COUNT = 5
SHUFFLE_SEED = 1337
BLIND_SEED = 4242


def load_sample() -> list[dict]:
    entries = []
    with SAMPLE_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_pipeline_results() -> dict[str, dict]:
    """Mappt note_name → latest result-dict für AGENT_VERSION."""
    results: dict[str, dict] = {}
    if not HISTORY_PATH.exists():
        return results
    with HISTORY_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("version") != AGENT_VERSION:
                continue
            name = Path(entry.get("note", "")).name
            if not name:
                continue
            results[name] = entry  # overrides older entries for same name
    return results


def trim_context(text: str, radius: int = CONTEXT_RADIUS) -> str:
    """Trimmt Kontext auf ±radius chars um die Mitte; ergänzt … bei Cut."""
    if len(text) <= 2 * radius:
        return text.strip()
    mid = len(text) // 2
    start = max(0, mid - radius)
    end = min(len(text), mid + radius)
    snippet = text[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def escape_md_cell(text: str) -> str:
    """Markdown-Tabellen-Zelle: pipe, newlines, backslash-escaping."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()


def slugify(text: str) -> str:
    text = re.sub(r"^(inbox|vault)__", "", text)
    text = re.sub(r"\.md$", "", text)
    text = re.sub(r"[^\w\säöüÄÖÜß-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text[:60].rstrip("-")


def render_note_md(idx: int, sample: dict, result: dict, *, blind: bool, total: int) -> str:
    note_name = sample["note_name"]
    pair = sample["language_pair"]
    pdf_path = sample["pdf_path"]
    claim_scores = result.get("claim_scores", [])
    mode_tag = "BLIND" if blind else "HYBRID"

    lines: list[str] = []
    lines.append(f"# Label {idx:02d}/{total} [{mode_tag}] — {note_name}")
    lines.append("")
    lines.append(f"- **Sprachpaar**: `{pair}`")
    lines.append(f"- **PDF**: `{pdf_path}`")
    lines.append(f"- **Folder**: `{sample['folder']}`")
    lines.append(f"- **Claims total**: {len(claim_scores)}")
    lines.append("")
    if blind:
        lines.append("## BLIND-Modus (Härtung #3)")
        lines.append("- **Kein Kontext-Snippet** wird gezeigt. Quelle: ausschließlich volle PDF.")
        lines.append("- Öffne die PDF und suche eigenständig nach Belegen für jeden Claim.")
        lines.append(
            "- Diese Labels gehen in `labels_blind.jsonl` und werden für Δκ-Kontrolle vs. Hybrid-Labels gerechnet."
        )
    else:
        lines.append("## Härtungs-Reminder")
        lines.append("- **Härtung #1**: Pipeline-Verdict ist NICHT sichtbar — nur Claim + Kontext.")
        lines.append(
            "- **Härtung #2**: Wenn Kontext den Claim NICHT stützt → STOP, PDF auf Seite öffnen, Voll-Suche, erst dann `h`."
        )
        lines.append("- **Snippet-Reihenfolge**: zufällig gemischt (kein Cosine-Ranking sichtbar).")
        lines.append("- **`?`-Regel**: nur bei genuiner Mehrdeutigkeit (siehe `labeling_protocol.md`).")
    lines.append("")
    lines.append("## Claims")
    lines.append("")

    if not claim_scores:
        lines.append("_Keine Claims im Pipeline-Output (möglicherweise leere Note oder Eval-Fehler)._")
        return "\n".join(lines) + "\n"

    shuffle_rng = random.Random(SHUFFLE_SEED + idx)
    for i, score in enumerate(claim_scores):
        claim_text = score.get("claim", "").strip()
        page = score.get("best_page")

        lines.append(f"### Claim {i + 1}")
        lines.append("")
        lines.append(f"> {claim_text}")
        lines.append("")
        if page is not None:
            lines.append(f"**PDF-Seite (Pipeline-Top-Match, nur als PDF-Navigation)**: {page}")
            lines.append("")
        else:
            lines.append("**PDF-Seite**: _nicht zugewiesen_")
            lines.append("")

        if not blind:
            contexts = list(score.get("retrieved_contexts", []))[:TOP_K_CONTEXTS]
            shuffle_rng.shuffle(contexts)
            lines.append("**Kontext-Snippets (Reihenfolge gemischt):**")
            lines.append("")
            for ctx in contexts:
                pages = ctx.get("pages", []) or []
                page_str = f"S. {pages[0]}" if pages else "S. ?"
                snippet = trim_context(ctx.get("text", ""))
                lines.append(f"- ({page_str}) {snippet}")
            lines.append("")
        else:
            lines.append("_(keine Snippets — Blind-Modus)_")
            lines.append("")

        lines.append("| Label (s/h/?) | Tag (optional) | Notiz |")
        lines.append("|---|---|---|")
        lines.append(f"| <!--claim_idx={i}--> | | |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "**Beim Speichern**: Label-Spalte in jeder Tabelle ausfüllen (`s`, `h`, oder `?`). `collect.py` parsed später."
    )
    return "\n".join(lines) + "\n"


def render_index(samples: list[dict], missing: list[str], blind_names: list[str]) -> str:
    lines = [
        "# Labeling-Index",
        "",
        "Reihenfolge: Sprachpaar gemischt (siehe `labeling_protocol.md` §Praktische Hinweise).",
        "",
    ]
    lines.append(f"- **Total Notes (Hybrid)**: {len(samples)}")
    lines.append(f"- **Blind-Sub-Sample**: {len(blind_names)} (in `../blind/`)")
    lines.append(f"- **Pipeline-Eval fehlend**: {len(missing)} (siehe Liste unten)")
    lines.append("")
    lines.append("## Hybrid-Notes")
    lines.append("")
    lines.append("| # | Sprachpaar | Note | Auch im Blind-Set? | Status |")
    lines.append("|---|---|---|---|---|")
    for idx, s in enumerate(samples, start=1):
        slug = slugify(s["note_name"])
        link = f"[{s['note_name']}](./{idx:02d}__{slug}.md)"
        in_blind = "🔁 ja" if s["note_name"] in blind_names else ""
        status = "❌ kein Pipeline-Output" if s["note_name"] in missing else "🟡 to-label"
        lines.append(f"| {idx:02d} | {s['language_pair']} | {link} | {in_blind} | {status} |")
    lines.append("")
    if blind_names:
        lines.append("## Blind-Sub-Sample (Härtung #3)")
        lines.append("")
        lines.append(
            "Dieselben Notes wie oben markiert, aber ohne Kontext-Snippets. Labels gehen separat in `labels_blind.jsonl`."
        )
        lines.append("")
        for b_idx, name in enumerate(blind_names, start=1):
            slug = slugify(name)
            lines.append(f"- [{b_idx:02d}__{slug}.md](../blind/{b_idx:02d}__{slug}.md) — {name}")
        lines.append("")
    if missing:
        lines.append("## Fehlende Pipeline-Outputs")
        lines.append("")
        lines.append("Diese Notes wurden gesampelt, aber `run.py` hat dafür noch keine Eval geschrieben:")
        for name in missing:
            lines.append(f"- `{name}`")
        lines.append("")
        lines.append("→ `run.py` (erneut) laufen lassen oder Notes aus dem Sample entfernen.")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not SAMPLE_FILE.exists():
        print(f"FEHLER: {SAMPLE_FILE} fehlt — erst sample.py laufen", file=sys.stderr)
        sys.exit(1)

    samples = load_sample()
    pipeline = load_pipeline_results()
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    BLIND_DIR.mkdir(parents=True, exist_ok=True)

    total = len(samples)
    missing: list[str] = []
    written = 0
    for idx, sample in enumerate(samples, start=1):
        note_name = sample["note_name"]
        result = pipeline.get(note_name)
        slug = slugify(note_name)
        out_path = NOTES_DIR / f"{idx:02d}__{slug}.md"

        if result is None:
            missing.append(note_name)
            placeholder = (
                f"# Label {idx:02d}/{total} — {note_name}\n\n"
                f"**Status**: kein Pipeline-Output (v{AGENT_VERSION}) in `quality_history.jsonl`.\n"
                f"→ erst `run.py` laufen lassen.\n"
            )
            out_path.write_text(placeholder, encoding="utf-8")
        else:
            out_path.write_text(
                render_note_md(idx, sample, result, blind=False, total=total),
                encoding="utf-8",
            )
            written += 1

    # Blind-Sub-Sample (Härtung #3): BLIND_COUNT zufällige Notes deterministisch wählen,
    # nur unter denen mit existierendem Pipeline-Output.
    available = [s for s in samples if s["note_name"] in pipeline]
    blind_rng = random.Random(BLIND_SEED)
    blind_pick = blind_rng.sample(available, min(BLIND_COUNT, len(available))) if available else []
    blind_names: list[str] = []
    for b_idx, sample in enumerate(blind_pick, start=1):
        slug = slugify(sample["note_name"])
        out_path = BLIND_DIR / f"{b_idx:02d}__{slug}.md"
        out_path.write_text(
            render_note_md(b_idx, sample, pipeline[sample["note_name"]], blind=True, total=len(blind_pick)),
            encoding="utf-8",
        )
        blind_names.append(sample["note_name"])

    index_path = NOTES_DIR / "INDEX.md"
    index_path.write_text(render_index(samples, missing, blind_names), encoding="utf-8")

    print(f"Generiert: {written}/{len(samples)} Hybrid-Label-Files → {NOTES_DIR}")
    print(f"           {len(blind_pick)} Blind-Label-Files → {BLIND_DIR}")
    if missing:
        print(f"  Fehlend (kein Pipeline-Output): {len(missing)}")
        for name in missing[:5]:
            print(f"    - {name}")
        if len(missing) > 5:
            print(f"    … +{len(missing) - 5} weitere")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
