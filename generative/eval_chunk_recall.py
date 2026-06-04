"""Deterministische Phase-A-Mechanik-Messung für Chunking-/Planner-Recall.

KEIN LLM-Call. Misst Retrieval-Eigenschaften der deterministischen Pipeline-
Funktionen, um zu entscheiden, ob Overlap-Chunking bzw. ein größeres Overview-
Budget einen messbaren Recall-Gewinn bringen — bevor teure stochastische
Pipeline-Läufe gestartet werden (pdfplumber-Lektion: erst rauschfrei messen).

Metriken:
  straddle_stats   — Sätze die von Wort-Split-Chunk-Grenzen zerschnitten werden.
                     Overlap-K holt einen Satz zurück, wenn er in ein um K Wörter
                     verlängertes Nachbar-Chunk vollständig passt.
"""
from __future__ import annotations

import re

_SENT_END_RE = re.compile(r"[.!?]$")
_PAGE_MARKER_RE = re.compile(r"\[S\.\s*\d+\]")


def _sentence_spans(words: list[str]) -> list[tuple[int, int]]:
    """[(start_word, end_word_exklusiv), ...] — Satz endet nach Wort mit . ! ?."""
    spans: list[tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        if _SENT_END_RE.search(w):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def _chunk_spans(n_words: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Wort-Split-Chunks als (start, end_exklusiv). stride = size - overlap."""
    stride = max(1, size - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_words:
        spans.append((start, min(start + size, n_words)))
        if start + size >= n_words:
            break
        start += stride
    return spans


_COVERAGE_THRESHOLDS = (0.5, 0.75, 1.0)


def overview_coverage(text: str, concept_titles: list[str]) -> dict:
    """Misst, welche Konzepte ihre Titel-Evidenz im extract_overview()-Output behalten
    — eine SCHWACHE, token-basierte Obergrenze für den Default-Modus-Planner-Recall
    (er kann nur planen, was im Overview steht).

    WICHTIG (Codex-Review 2026-06-04): Die Metrik ist token-basiert und nutzt dieselbe
    `_tokens`-Logik wie `planner.filter_hallucinated` — sie misst „würde ein Titel-
    Token-Filter anschlagen", NICHT semantische Präsenz. Eine einzelne Coverage-Schwelle
    (vorher 0.5) erzeugt Schwellen-Artefakte (ein generisches Token reicht bei
    Zwei-Token-Titeln). Deshalb wird eine Sensitivitätskurve über mehrere Schwellen
    plus exakter Phrasen-Match berichtet, nie eine Einzelzahl.

    Coverage = |Titel-Tokens ∩ Output-Tokens| / |Titel-Tokens|. `in_fulltext` =
    Coverage gegen Volltext ≥ 0.5 (Konzept überhaupt im Dokument). Da der Overview aus
    dem Volltext gebaut ist (Output-Tokens ⊆ Volltext-Tokens), gilt cov_ov ≤ cov_full.

    Returns {n_concepts, n_in_fulltext, covered, exact_phrase, missed_strict}:
      covered       — {schwelle: Anzahl in_fulltext-Konzepte mit cov_ov ≥ schwelle}
      exact_phrase  — Konzepte, deren normalisierter Titel als Substring im Overview steht
      missed_strict — in_fulltext-Konzepte mit cov_ov < 1.0 (kippen bei strenger Schwelle)
    """
    from agents.cross_reference import _tokens
    from pipeline.pdf_chunker import extract_overview

    overview = extract_overview(text)
    overview_tokens = _tokens(overview)
    overview_norm = " ".join(overview.lower().split())
    full_tokens = _tokens(text)

    n_in_fulltext = 0
    covered = {t: 0 for t in _COVERAGE_THRESHOLDS}
    exact_phrase = 0
    missed_strict: list[str] = []
    for title in concept_titles:
        title_tokens = _tokens(title)
        if not title_tokens:
            continue
        cov_full = len(title_tokens & full_tokens) / len(title_tokens)
        if cov_full < 0.5:
            continue  # Konzept nicht im Dokument — irrelevant für Recall-Decke
        n_in_fulltext += 1
        cov_ov = len(title_tokens & overview_tokens) / len(title_tokens)
        for t in _COVERAGE_THRESHOLDS:
            if cov_ov >= t:
                covered[t] += 1
        if " ".join(title.lower().split()) in overview_norm:
            exact_phrase += 1
        if cov_ov < 1.0:
            missed_strict.append(title)
    return {
        "n_concepts": len(concept_titles),
        "n_in_fulltext": n_in_fulltext,
        "covered": covered,
        "exact_phrase": exact_phrase,
        "missed_strict": missed_strict,
    }


def straddle_stats(text: str, size: int, overlap: int = 0) -> dict:
    """Zählt Sätze die von keinem einzelnen Chunk vollständig erfasst werden.

    Misst den `_split_by_words`-Pfad (fester `size`, kein Overlap), NICHT den Default
    `split_by_chapters`/`concept_text_window` (Codex-Finding 2026-06-04) — daher
    `mode="word_split"` im Output.

    Returns {n_sentences, n_straddling, mode}. Ein Satz straddelt, wenn kein Chunk-Span
    ihn komplett enthält — bei overlap=0 sind das alle satzgrenzen-überschreitenden
    Cuts, mit overlap>0 werden randnahe Sätze zurückgeholt.

    `[S. N]`-Seitenmarker werden vor der Satzsegmentierung entfernt — sonst erzeugt
    das Token `[S.` (endet auf `.`) Pseudo-Sätze und verfälscht die Rate.
    """
    if size <= 0 or not (0 <= overlap < size):
        raise ValueError(f"size>0 und 0<=overlap<size erforderlich, war size={size}, overlap={overlap}")
    words = _PAGE_MARKER_RE.sub(" ", text).split()
    sentences = _sentence_spans(words)
    chunks = _chunk_spans(len(words), size, overlap)
    n_straddling = 0
    for s_start, s_end in sentences:
        contained = any(c_start <= s_start and s_end <= c_end for c_start, c_end in chunks)
        if not contained:
            n_straddling += 1
    return {"n_sentences": len(sentences), "n_straddling": n_straddling,
            "mode": "word_split"}


# ---- Runner (LLM-frei) ------------------------------------------------------

_LABEL_TITLE_RE = re.compile(
    r"^#\s*Label\b.*?—\s*(?:vault__|inbox__|merge__)?(.+?)\.md\s*$", re.MULTILINE)
_LABEL_PDF_RE = re.compile(r"\*\*PDF\*\*:\s*`?([^`\n]+?)`?\s*$", re.MULTILINE)


def _label_concept(text: str) -> tuple[str | None, str | None]:
    """Parst (PDF-Pfad, Konzept-Titel) aus einer calibration/labels-active-Note."""
    t = _LABEL_TITLE_RE.search(text)
    p = _LABEL_PDF_RE.search(text)
    return (p.group(1).strip() if p else None,
            t.group(1).strip() if t else None)


def concepts_from_label_notes(label_dir, pdf_substring: str) -> list[str]:
    """Menschlich kuratierte Konzept-Titel für eine Quelle (source-korrekt, gold-grade).

    Liest alle Label-Notes in `label_dir` und gibt die Titel zurück, deren `**PDF**`
    den Teilstring `pdf_substring` enthält. Nicht-zirkuläre Referenz für
    overview_coverage (stammt aus menschlichem Labeling, nicht aus dem Overview).
    """
    from pathlib import Path
    titles: list[str] = []
    for f in sorted(Path(label_dir).glob("*.md")):
        if f.name == "INDEX.md":
            continue
        pdf, title = _label_concept(f.read_text(encoding="utf-8"))
        if title and pdf and pdf_substring.lower() in pdf.lower():
            titles.append(title)
    return titles


def _concepts_from_baseline(baseline_dir) -> list[str]:
    """Konzept-Titel aus den Baseline-Cache-Notes als Referenzmenge (gold-frei)."""
    from pathlib import Path
    titles: list[str] = []
    for f in sorted(Path(baseline_dir).glob("*.md")):
        name = f.stem
        for pfx in ("vault__", "merge__"):
            if name.startswith(pfx):
                name = name[len(pfx):]
                break
        name = re.sub(r"^MERGE\s*-\s*", "", name)
        titles.append(name)
    return titles


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path

    from config import CHUNK_WORDS
    from pipeline.pdf_chunker import pdf_to_text

    p = argparse.ArgumentParser(description="Phase-A Chunk-/Planner-Recall-Messung (LLM-frei)")
    p.add_argument("pdf", type=Path, help="Pfad zum Quell-PDF")
    p.add_argument("--concepts", action="append", default=None,
                   help="Konzept-Titel (mehrfach). Default: aus --baseline-dir.")
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="Baseline-Cache-Ordner mit *.md als Konzept-Referenz.")
    p.add_argument("--label-dir", type=Path, default=None,
                   help="calibration/labels-active-Ordner als (source-korrekte) Konzept-Referenz.")
    p.add_argument("--pdf-match", default=None,
                   help="Teilstring zum Filtern der Label-Notes nach **PDF** (z.B. 'Porst - 2014').")
    p.add_argument("--overlaps", type=int, nargs="+", default=[0, 50, 100, 200],
                   help="Zu testende Overlap-Wortzahlen.")
    args = p.parse_args(argv)

    if not args.pdf.exists():
        print(f"PDF nicht gefunden: {args.pdf}", file=sys.stderr)
        return 1

    text = pdf_to_text(args.pdf)
    n_words = len(text.split())
    print(f"# {args.pdf.name}  ({n_words} Wörter, CHUNK_WORDS={CHUNK_WORDS})\n")

    print("## Boundary-Loss (Wort-Split, kein Kapitel-Fallback)")
    base = straddle_stats(text, size=CHUNK_WORDS, overlap=0)
    print(f"  Sätze gesamt: {base['n_sentences']}")
    for ov in args.overlaps:
        st = straddle_stats(text, size=CHUNK_WORDS, overlap=ov)
        n = st["n_straddling"]
        pct = 100 * n / st["n_sentences"] if st["n_sentences"] else 0
        recovered = base["n_straddling"] - n
        print(f"  overlap={ov:>4}: straddling={n:>4} ({pct:4.1f}%)  zurückgeholt ggü. ov=0: {recovered}")

    concepts = args.concepts
    if not concepts and args.label_dir and args.pdf_match:
        concepts = concepts_from_label_notes(args.label_dir, args.pdf_match)
    if not concepts and args.baseline_dir:
        concepts = _concepts_from_baseline(args.baseline_dir)
    if concepts:
        print("\n## Overview-Coverage (schwache token-basierte Recall-Obergrenze)")
        cov = overview_coverage(text, concepts)
        n = cov["n_in_fulltext"]
        print(f"  Konzepte: {cov['n_concepts']} | im Volltext: {n}")
        for t in _COVERAGE_THRESHOLDS:
            print(f"    covered @ coverage≥{t}: {cov['covered'][t]}/{n}")
        print(f"    exakter Phrasen-Match:    {cov['exact_phrase']}/{n}")
        if cov["missed_strict"]:
            print("  Kippt bei strenger Schwelle (cov<1.0): " + ", ".join(cov["missed_strict"]))
    else:
        print("\n## Overview-Coverage übersprungen (keine --concepts / --baseline-dir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
