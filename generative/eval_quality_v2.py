#!/usr/bin/env python3
"""Claim-zentrierte Qualitätsmessung via Retrieval + mDeBERTa-NLI.

eval_quality_v2.py läuft parallel zur alten eval_quality.py. Die v2-Metriken
bewerten atomare Claims aus dem Note-Body statt source_anchors und nutzen NLI
als Entscheidungsbasis für Paraphrasen.

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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF fehlt: pip install pymupdf")

from generative.config import AGENT_VERSION, CACHE_DIR, MDEBERTA_NLI_MODEL
from generative.eval_quality import _extract_page_text, _normalize, wilson_ci
from generative.pipeline.embeddings import _model, cosine

_QUALITY_HISTORY = CACHE_DIR / "quality_history.jsonl"
EVAL_VERSION = "2.0"

TOP_K = 5
CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 180
EXPANSION_MAX_TOKENS = 450

_NLI_CACHE: dict[str, object] = {}

_ABBREVIATIONS = [
    "z.B.", "d.h.", "bzw.", "vgl.", "bspw.", "usw.", "etc.", "ggf.",
    "Hrsg.", "Jg.", "Bd.", "S.", "Nr.", "Abb.", "Tab.", "Kap.", "ff.",
]

_CAPTION_RE = re.compile(r"^\s*(Abb\.|Abbildung|Fig\.|Figure|Tab\.|Tabelle)\s*\d+[:.]", re.I)
_FOOTNOTE_DEF_RE = re.compile(r"^\[\^\d+\]:.*$", re.MULTILINE)
_FOOTNOTE_MARKER_RE = re.compile(r"\[\^\d+\]")
_BIB_START_RE = re.compile(r"^\s*(literatur|bibliographie|references|bibliography)\s*$", re.I)
_BIB_ENTRY_RE = re.compile(
    r"^\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'`-]+,\s+.*\b(19|20)\d{2}\b"
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    idx: int
    text: str
    pages: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.text.split())


def _read_note_body(note_path: Path) -> str:
    text = note_path.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        fm_end = text.find("\n---", 3)
        if fm_end != -1:
            text = text[text.find("\n", fm_end + 4) + 1:]

    title = re.search(r"^#\s+.+$", text, flags=re.MULTILINE)
    if title:
        text = text[title.start():]

    sources = re.search(r"^##\s+Quellen\b", text, flags=re.MULTILINE | re.I)
    if sources:
        text = text[:sources.start()]
    return text


def _drop_quote_callouts(markdown: str) -> str:
    lines = markdown.splitlines()
    kept: list[str] = []
    in_quote_callout = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("> [!quote]"):
            in_quote_callout = True
            continue
        if in_quote_callout:
            if stripped.startswith(">"):
                continue
            in_quote_callout = False
        kept.append(line)
    return "\n".join(kept)


def _protect_sentence_periods(text: str) -> str:
    sentinel = "__PERIOD__"
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", sentinel))

    text = re.sub(r"\b[A-Z]\.", lambda m: m.group(0).replace(".", sentinel), text)
    text = re.sub(r"\d+\.\d+", lambda m: m.group(0).replace(".", sentinel), text)
    text = re.sub(r"S\.\s*\d+(?:-\d+)?", lambda m: m.group(0).replace(".", sentinel), text)
    return text


def _split_sentences(text: str) -> list[str]:
    protected = _protect_sentence_periods(text)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [p.replace("__PERIOD__", ".").strip() for p in parts if p.strip()]


def extract_claims(note_path: Path) -> list[str]:
    """Extrahiert atomare Claims aus dem Note-Body."""
    body = _read_note_body(note_path)
    body = _drop_quote_callouts(body)
    body = _FOOTNOTE_DEF_RE.sub("", body)
    body = _FOOTNOTE_MARKER_RE.sub("", body)
    body = re.sub(r"^#+\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"[*_`>#-]+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    claims: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(body):
        sentence = sentence.strip(" \t\r\n:;,-")
        if len(sentence) < 30:
            continue
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        claims.append(sentence)
    return claims


def _raw_page_lines(pdf_doc: fitz.Document, page_num: int) -> list[str]:
    try:
        page = pdf_doc[page_num - 1]
    except IndexError:
        return []
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if len(b) > 6 and b[6] == 0]
    text_blocks.sort(key=lambda b: (round(float(b[1]) / 8) * 8, float(b[0])))
    lines: list[str] = []
    for block in text_blocks:
        for line in str(block[4]).splitlines():
            line = _normalize(line)
            if line:
                lines.append(line)
    return lines


def _recurring_short_lines(pages: list[list[str]]) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({line for line in lines if len(line) <= 80})
    threshold = max(3, int(len(pages) * 0.2))
    return {line for line, count in counts.items() if count >= threshold}


def _filter_boilerplate_pages(pages: list[list[str]]) -> list[list[str]]:
    recurring = _recurring_short_lines(pages)
    filtered_pages: list[list[str]] = []
    for lines in pages:
        bib_seen = False
        bib_hits = 0
        filtered: list[str] = []
        for line in lines:
            if line in recurring or _CAPTION_RE.match(line):
                continue
            if _BIB_START_RE.match(line):
                bib_seen = True
                continue
            if bib_seen:
                if _BIB_ENTRY_RE.match(line):
                    bib_hits += 1
                if bib_hits >= 3 or _BIB_ENTRY_RE.match(line):
                    continue
            filtered.append(line)
        filtered_pages.append(filtered)
    return filtered_pages


def _pdf_sentences(pdf_doc: fitz.Document) -> list[tuple[str, int]]:
    raw_pages = [_raw_page_lines(pdf_doc, page) for page in range(1, len(pdf_doc) + 1)]
    if not any(raw_pages):
        raw_pages = [[_extract_page_text(pdf_doc, page)] for page in range(1, len(pdf_doc) + 1)]
    pages = _filter_boilerplate_pages(raw_pages)

    sentences: list[tuple[str, int]] = []
    for page_num, lines in enumerate(pages, start=1):
        text = _normalize(" ".join(lines))
        for sentence in _split_sentences(text):
            sentence = sentence.strip()
            if len(sentence) >= 20:
                sentences.append((sentence, page_num))
    return sentences


def build_chunks(pdf_path: Path) -> list[Chunk]:
    with fitz.open(str(pdf_path)) as pdf_doc:
        sentences = _pdf_sentences(pdf_doc)

    chunks: list[Chunk] = []
    current: list[str] = []
    pages: set[int] = set()
    token_count = 0

    for sentence, page in sentences:
        sent_tokens = len(sentence.split())
        if current and token_count + sent_tokens > CHUNK_MAX_TOKENS and token_count >= CHUNK_MIN_TOKENS:
            chunks.append(Chunk(len(chunks), _normalize(" ".join(current)), tuple(sorted(pages))))
            overlap = current[-1:]
            current = overlap[:]
            pages = {page}
            token_count = sum(len(s.split()) for s in current)

        current.append(sentence)
        pages.add(page)
        token_count += sent_tokens

    if current:
        chunks.append(Chunk(len(chunks), _normalize(" ".join(current)), tuple(sorted(pages))))
    return chunks


def _expand_context(chunks: list[Chunk], idx: int) -> str:
    selected = [i for i in (idx - 1, idx, idx + 1) if 0 <= i < len(chunks)]
    words = " ".join(chunks[i].text for i in selected).split()
    if len(words) > EXPANSION_MAX_TOKENS:
        return " ".join(words[:EXPANSION_MAX_TOKENS])
    return " ".join(words)


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
        return [
            {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
            for _ in premises
        ]

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


def _detect_language_pair(note_text: str, pdf_text: str) -> str:
    umlaut_re = re.compile(r"[äöüßÄÖÜ]")
    de_words_re = re.compile(r"\b(der|die|das|und|ist|werden|nicht|eine|einer)\b", re.I)
    en_words_re = re.compile(r"\b(the|and|is|are|not|with|from|this|that)\b", re.I)

    def lang(sample: str) -> str:
        if umlaut_re.search(sample) or len(de_words_re.findall(sample[:2000])) >= len(en_words_re.findall(sample[:2000])):
            return "DE"
        return "EN"

    return f"{lang(pdf_text)}→{lang(note_text)}"


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

        scores.append({
            "claim": claim,
            "entailment": round(max_entailment, 3),
            "contradiction": round(max_contradiction, 3),
            "neutral": round(max_neutral, 3),
            "label": label,
            "best_chunk_idx": best_chunk_idx,
            "best_page": best_chunk.pages[0] if best_chunk.pages else None,
        })
    return scores


def eval_note(note_path: Path | str, pdf_path: Path | str, pipeline_version: str = AGENT_VERSION,
              no_cache: bool = False) -> dict:
    """Evaluiert eine Note gegen ihre Quell-PDF und gibt v2-Metriken zurück."""
    del no_cache  # Akzeptiert für CLI-Kompatibilität; v2 nutzt keinen Disk-Cache.
    note_path = Path(note_path)
    pdf_path = Path(pdf_path)
    timestamp = datetime.now().isoformat()

    if not note_path.exists():
        return {
            "note": note_path.name, "pdf": pdf_path.name, "version": pipeline_version,
            "eval_version": EVAL_VERSION, "timestamp": timestamp,
            "error": "note_not_found", "claims_total": 0,
        }
    if not pdf_path.exists():
        return {
            "note": note_path.name, "pdf": pdf_path.name, "version": pipeline_version,
            "eval_version": EVAL_VERSION, "timestamp": timestamp,
            "error": "pdf_not_found", "claims_total": 0,
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
    mean_entailment = (
        sum(score["entailment"] for score in claim_scores) / total if total else 0.0
    )
    low_entailment = sum(1 for score in claim_scores if score["entailment"] < 0.70)
    confirmed_chunks = {
        score["best_chunk_idx"] for score in claim_scores
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
