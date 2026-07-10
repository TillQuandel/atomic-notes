#!/usr/bin/env python3
"""Claim-zentrierte Qualitätsmessung via Retrieval + mDeBERTa-NLI.

eval_quality_v2.py läuft parallel zur alten eval_quality.py. Die v2-Metriken
bewerten atomare Claims aus dem Note-Body statt source_anchors und nutzen NLI
als Entscheidungsbasis für Paraphrasen.

STATUS (#98): historischer Runner, kein toter Code. Ist gleichzeitig die
Utility-Bibliothek fuer v4 UND traegt einen eigenen Claim-Scoring-Runner
(_nli_batch/_score_claims/eval_note hier), der bei calibration/run.py und
calibration/adversarial.py NICHT mehr aufgerufen wird (die laufen gegen
eval_quality_v4) — build_chunks()/eval_note() hier sind nur noch direkt
getestet (test_eval_page_namespace.py, test_stage8_pdf_memo.py), nicht mehr
produktiv verdrahtet. NICHT löschen (Tests haengen dran). Geteilte Utilities
(Chunk, TOP_K, _read_note_body, _detect_language_pair, _expand_context,
_chunks_from_sentences, _pdf_sentences, extract_claims, filter_meta_claims),
die auch v4/calibration nutzen, leben seit #98 in eval_common.py — hier nur
re-exportiert, damit dieser Runner und bestehende Tests unveraendert laufen.

Bekannte Failure-Modes:
- Tabellen-/Grafik-bezogene Claims können false-positive als halluziniert enden.
- Quer-Satz-Synthesen aus weit entfernten PDF-Stellen können false-positive sein.
- Mehrspaltige PDFs hängen weiter von der Block-Reihenfolge der PDF-Extraktion ab.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF fehlt: pip install pymupdf")

from generative.config import AGENT_VERSION, CACHE_DIR, MDEBERTA_NLI_MODEL

# #98: geteilte Utilities nach eval_common verschoben, hier re-exportiert --
# build_chunks/_score_claims/eval_note unten nutzen sie unveraendert per Re-Import;
# filter_meta_claims wird von keinem Code hier mehr direkt gebraucht (extract_claims
# ruft es intern in eval_common auf), bleibt aber re-exportiert fuer
# test_claim_meta_filter.py (direkter Import aus diesem Modul).
from generative.eval_common import (
    Chunk,
    TOP_K,
    _detect_language_pair,
    _expand_context,
    _chunks_from_sentences,
    _pdf_sentences,
    _read_note_body,
    extract_claims,
    filter_meta_claims,  # noqa: F401  # re-export, extern genutzt (test_claim_meta_filter)
    wilson_ci,
)
from generative.embeddings import _model, cosine
from generative.pipeline.pdf_chunker import anchor_page_numbers

_QUALITY_HISTORY = CACHE_DIR / "quality_history.jsonl"
EVAL_VERSION = "2.0"

_NLI_CACHE: dict[str, object] = {}

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def build_chunks(pdf_path: Path) -> list[Chunk]:
    with fitz.open(str(pdf_path)) as pdf_doc:
        page_numbers = anchor_page_numbers(pdf_path, len(pdf_doc))
        sentences = _pdf_sentences(pdf_doc, page_numbers)
    return _chunks_from_sentences(sentences)


def _nli_model():
    if "loaded" in _NLI_CACHE:
        return _NLI_CACHE.get("tokenizer"), _NLI_CACHE.get("model"), _NLI_CACHE.get("torch")
    _NLI_CACHE["loaded"] = True
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MDEBERTA_NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MDEBERTA_NLI_MODEL)
        model.eval()
        _NLI_CACHE.update({"tokenizer": tokenizer, "model": model, "torch": torch})
        return tokenizer, model, torch
    except Exception as exc:
        print(f"  [mDeBERTa] Modell nicht geladen: {exc}", file=sys.stderr)
        _NLI_CACHE.update({"tokenizer": None, "model": None, "torch": None})
        return None, None, None


def _label_index_map(model) -> dict[str, int]:
    labels = getattr(model.config, "id2label", {}) or {}
    mapped: dict[str, int] = {}
    for idx, label in labels.items():
        mapped[str(label).lower()] = int(idx)
    return mapped


def _nli_batch(premises: list[str], hypothesis: str) -> list[dict[str, float]]:
    tokenizer, model, torch = _nli_model()
    if tokenizer is None or model is None or torch is None:
        return [{"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0} for _ in premises]

    inputs = tokenizer(
        premises,
        [hypothesis] * len(premises),
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).tolist()
    label_map = _label_index_map(model)
    e_idx = label_map.get("entailment", 0)
    n_idx = label_map.get("neutral", 1)
    c_idx = label_map.get("contradiction", 2)
    return [
        {
            "entailment": float(row[e_idx]),
            "neutral": float(row[n_idx]),
            "contradiction": float(row[c_idx]),
        }
        for row in probs
    ]


def _label_claim(entailment: float, contradiction: float) -> str:
    if contradiction >= 0.50 and entailment < 0.40:
        return "contradiction"
    if entailment >= 0.70:
        return "confirmed"
    if entailment >= 0.40:
        return "uncertain"
    return "hallucinated"


def _score_claims(claims: list[str], chunks: list[Chunk]) -> list[dict]:
    if not claims or not chunks:
        return []

    model = _model()
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_embs = model.encode(chunk_texts, show_progress_bar=False, normalize_embeddings=True)
    claim_embs = model.encode(claims, show_progress_bar=False, normalize_embeddings=True)

    scores: list[dict] = []
    for claim, claim_emb in zip(claims, claim_embs):
        ranked = sorted(
            ((idx, cosine(chunk_embs[idx], claim_emb)) for idx in range(len(chunks))),
            key=lambda item: item[1],
            reverse=True,
        )[:TOP_K]
        contexts = [_expand_context(chunks, idx) for idx, _ in ranked]
        nli_scores = _nli_batch(contexts, claim)

        best_entailment_idx = max(range(len(nli_scores)), key=lambda i: nli_scores[i]["entailment"])
        max_entailment = nli_scores[best_entailment_idx]["entailment"]
        max_contradiction = max(s["contradiction"] for s in nli_scores)
        max_neutral = max(s["neutral"] for s in nli_scores)
        best_chunk_idx = ranked[best_entailment_idx][0]
        best_chunk = chunks[best_chunk_idx]
        label = _label_claim(max_entailment, max_contradiction)

        scores.append(
            {
                "claim": claim,
                "entailment": round(max_entailment, 3),
                "contradiction": round(max_contradiction, 3),
                "neutral": round(max_neutral, 3),
                "label": label,
                "best_chunk_idx": best_chunk_idx,
                "best_page": best_chunk.pages[0] if best_chunk.pages else None,
            }
        )
    return scores


def eval_note(
    note_path: Path | str, pdf_path: Path | str, pipeline_version: str = AGENT_VERSION, no_cache: bool = False
) -> dict:
    """Evaluiert eine Note gegen ihre Quell-PDF und gibt v2-Metriken zurück."""
    del no_cache  # Akzeptiert für CLI-Kompatibilität; v2 nutzt keinen Disk-Cache.
    note_path = Path(note_path)
    pdf_path = Path(pdf_path)
    timestamp = datetime.now().isoformat()

    if not note_path.exists():
        return {
            "note": note_path.name,
            "pdf": pdf_path.name,
            "version": pipeline_version,
            "eval_version": EVAL_VERSION,
            "timestamp": timestamp,
            "error": "note_not_found",
            "claims_total": 0,
        }
    if not pdf_path.exists():
        return {
            "note": note_path.name,
            "pdf": pdf_path.name,
            "version": pipeline_version,
            "eval_version": EVAL_VERSION,
            "timestamp": timestamp,
            "error": "pdf_not_found",
            "claims_total": 0,
        }

    note_body = _read_note_body(note_path)
    claims = extract_claims(note_path)
    chunks = build_chunks(pdf_path)
    pdf_sample = chunks[0].text if chunks else ""
    language_pair = _detect_language_pair(note_body, pdf_sample)

    claim_scores = _score_claims(claims, chunks) if claims and chunks else []
    total = len(claims)
    confirmed = sum(1 for score in claim_scores if score["label"] == "confirmed")
    uncertain = sum(1 for score in claim_scores if score["label"] == "uncertain")
    hallucinated = sum(1 for score in claim_scores if score["label"] == "hallucinated")
    contradiction = sum(1 for score in claim_scores if score["label"] == "contradiction")
    support_rate = confirmed / total if total else 0.0
    hallucination_rate = (hallucinated + contradiction) / total if total else 0.0
    mean_entailment = sum(score["entailment"] for score in claim_scores) / total if total else 0.0
    low_entailment = sum(1 for score in claim_scores if score["entailment"] < 0.70)
    confirmed_chunks = {
        score["best_chunk_idx"]
        for score in claim_scores
        if score["label"] == "confirmed" and score["best_chunk_idx"] is not None
    }
    source_span_diversity = len(confirmed_chunks) / len(chunks) if chunks else 0.0

    result = {
        "note": note_path.name,
        "pdf": pdf_path.name,
        "language": language_pair,
        "version": pipeline_version,
        "eval_version": EVAL_VERSION,
        "timestamp": timestamp,
        "claims_total": total,
        "claims_confirmed": confirmed,
        "claims_uncertain": uncertain,
        "claims_hallucinated": hallucinated,
        "claims_contradiction": contradiction,
        "claim_support_rate": round(support_rate, 3),
        "mean_entailment": round(mean_entailment, 3),
        "max_entailment_below_threshold_count": low_entailment,
        "source_span_diversity": round(source_span_diversity, 3),
        "anchors_total": total,
        "anchors_confirmed": confirmed,
        "anchors_uncertain": uncertain,
        "anchors_hallucinated": hallucinated + contradiction,
        "hallucination_rate": round(hallucination_rate, 3),
        "coverage_rate": round(support_rate, 3),
        "hallucination_ci_95": wilson_ci(hallucinated + contradiction, total),
        "pdf_chunks_total": len(chunks),
        "claim_scores": claim_scores,
    }
    if not chunks:
        result["error"] = "pdf_not_parseable"
    elif not claims:
        result["error"] = "no_claims_found"
    return result


def save_result(result: dict) -> None:
    """Appended Ergebnis an quality_history.jsonl."""
    _QUALITY_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with _QUALITY_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"  -> gespeichert: {_QUALITY_HISTORY}")


def print_summary(result: dict) -> None:
    if "error" in result:
        print(f"[ERROR] {result['note']}: {result['error']}")
        return
    print(
        f"[eval_quality_v2] {result['note']}: "
        f"{result['claims_confirmed']}/{result['claims_total']} confirmed, "
        f"{result['claims_uncertain']} uncertain, "
        f"{result['claims_hallucinated']} hallucinated, "
        f"{result['claims_contradiction']} contradiction, "
        f"support={result['claim_support_rate']:.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-zentrierte NLI-Qualitätsmessung")
    parser.add_argument("--note", help="Pfad zur Note-Datei (.md)")
    parser.add_argument("--pdf", help="Pfad zur Quell-PDF")
    parser.add_argument("--version", default=AGENT_VERSION, help="Pipeline-Version")
    parser.add_argument("--save", action="store_true", help="Ergebnis in quality_history.jsonl speichern")
    parser.add_argument("--no-cache", action="store_true", help="Kompatibilitätsflag; v2 nutzt keinen Disk-Cache")
    args = parser.parse_args()

    if not args.note or not args.pdf:
        parser.print_help()
        sys.exit(1)

    result = eval_note(Path(args.note), Path(args.pdf), args.version, no_cache=args.no_cache)
    print_summary(result)
    if args.save:
        save_result(result)


if __name__ == "__main__":
    main()
