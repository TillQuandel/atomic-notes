from __future__ import annotations
import json, re
from datetime import datetime
from pathlib import Path
from rapidfuzz import fuzz

_ANCHOR_RE = re.compile(r"\s*\(S\.\s*\d+(?:-\d+)?\)")


def _strip(text: str) -> str:
    return _ANCHOR_RE.sub("", text).strip()


def compute_anchor_rate(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if _ANCHOR_RE.search(s)) / len(sentences)


def compute_hallucination_rate(sentences: list[str], fulltext: str, threshold: int = 75) -> float:
    """Anteil Saetze die NICHT im PDF-Volltext auffindbar sind (rapidfuzz).
    Saetze < 10 Zeichen werden uebersprungen (zu kurz fuer sinnvollen Match).
    Anker werden vor dem Vergleich entfernt.
    """
    if not sentences:
        return 0.0
    scorable = [s for s in sentences if len(_strip(s)) >= 12]
    if not scorable:
        return 0.0
    hallucinated = sum(
        1 for s in scorable
        if fuzz.partial_ratio(_strip(s).lower(), fulltext.lower()) < threshold
    )
    return hallucinated / len(scorable)


def save_eval_result(
    note_title: str,
    sentences: list[str],
    fulltext: str,
    source_file: str,
    pipeline_version: str,
    out_jsonl: Path,
) -> dict:
    result = {
        "note": note_title,
        "pdf": source_file,
        "pipeline": "foss-atomic",
        "version": pipeline_version,
        "eval_version": "foss-1.0",
        "hallucination_rate": compute_hallucination_rate(sentences, fulltext),
        "anchor_rate": compute_anchor_rate(sentences),
        "coverage_factual": None,  # Sentinel: nicht messbar in FOSS-Eval
        "timestamp": datetime.now().isoformat(),
    }
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result
