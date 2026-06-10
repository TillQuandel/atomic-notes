#!/usr/bin/env python3
"""LLM-as-Judge Qualitaetsmessung fuer Atomic-Agent Notes.

v3 bewertet atomare Claims gegen Top-K-PDF-Kontexte mit Claude als Judge,
verifiziert angegebene Zitate programmatisch und aggregiert Retrieval-Failures
separat von Halluzinationen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF fehlt: pip install pymupdf")

try:
    from rapidfuzz import fuzz
except ImportError:
    sys.exit("rapidfuzz fehlt: pip install rapidfuzz")

from generative.agents import base
from generative.config import AGENT_VERSION, CACHE_DIR, MODEL_OPUS
from generative.eval_quality import _extract_page_text, _normalize, wilson_ci
from generative.eval_quality_v2 import TOP_K, Chunk, _detect_language_pair, _expand_context, _read_note_body
from generative.eval_quality_v2 import build_chunks, extract_claims
from generative.pipeline.embeddings import _model, cosine

_QUALITY_HISTORY = CACHE_DIR / "quality_history.jsonl"
EVAL_VERSION = "3.2"

SUPPORTED_EXACT = "supported_exact"
SUPPORTED_PARAPHRASE = "supported_paraphrase"
PARTIALLY_SUPPORTED = "partially_supported"
NOT_IN_CONTEXT = "not_in_context"
CONTRADICTED = "contradicted"
RETRIEVAL_UNCERTAIN = "retrieval_or_parse_uncertain"
PARSE_ERROR = "parse_error"
JUDGE_LABELS = {
    SUPPORTED_EXACT,
    SUPPORTED_PARAPHRASE,
    PARTIALLY_SUPPORTED,
    NOT_IN_CONTEXT,
    CONTRADICTED,
}
LABELS = {
    *JUDGE_LABELS,
    RETRIEVAL_UNCERTAIN,
    PARSE_ERROR,
}
LABEL_STRICTNESS = {
    SUPPORTED_EXACT: 0,
    SUPPORTED_PARAPHRASE: 1,
    PARTIALLY_SUPPORTED: 2,
    NOT_IN_CONTEXT: 3,
    CONTRADICTED: 4,
}
SYSTEM_LABELS = {RETRIEVAL_UNCERTAIN, PARSE_ERROR}

RETRIEVAL_LOW_COSINE = 0.4
MAX_PROMPT_WORDS = 12000


@dataclass
class RetrievedContext:
    claim_idx: int
    claim: str
    contexts: list[dict[str, Any]]
    top_cosine: float
    best_chunk_idx: int | None
    best_page: int | None


def _note_title(note_path: Path, note_body: str) -> str:
    match = re.search(r"^#\s+(.+)$", note_body, flags=re.MULTILINE)
    return match.group(1).strip() if match else note_path.stem


def _retrieve_claim_contexts(claims: list[str], chunks: list[Chunk]) -> list[RetrievedContext]:
    if not claims or not chunks:
        return []

    model = _model()
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_embs = model.encode(chunk_texts, show_progress_bar=False, normalize_embeddings=True)
    claim_embs = model.encode(claims, show_progress_bar=False, normalize_embeddings=True)

    retrieved: list[RetrievedContext] = []
    for claim_idx, (claim, claim_emb) in enumerate(zip(claims, claim_embs)):
        ranked = sorted(
            ((idx, cosine(chunk_embs[idx], claim_emb)) for idx in range(len(chunks))),
            key=lambda item: item[1],
            reverse=True,
        )[:TOP_K]
        contexts: list[dict[str, Any]] = []
        for rank, (chunk_idx, score) in enumerate(ranked, start=1):
            chunk = chunks[chunk_idx]
            contexts.append({
                "rank": rank,
                "chunk_idx": chunk_idx,
                "pages": list(chunk.pages),
                "cosine": round(float(score), 3),
                "text": _expand_context(chunks, chunk_idx),
            })
        best_idx = ranked[0][0] if ranked else None
        best_chunk = chunks[best_idx] if best_idx is not None else None
        retrieved.append(RetrievedContext(
            claim_idx=claim_idx,
            claim=claim,
            contexts=contexts,
            top_cosine=round(float(ranked[0][1]), 3) if ranked else 0.0,
            best_chunk_idx=best_idx,
            best_page=best_chunk.pages[0] if best_chunk and best_chunk.pages else None,
        ))
    return retrieved


def _prompt_header(note_title: str, note_body: str, *, variant: str) -> str:
    if variant == "audit":
        stance = "Pruefe skeptisch, ob der Claim wirklich aus dem Kontext folgt. Bevorzuge strenge Labels bei Retrieval-Problemen."
    else:
        stance = "Pruefe fair, ob der Claim im Kontext exakt, sinngemaess, teilweise, nicht oder widerspruechlich belegt ist."
    compact_body = " ".join(note_body.split())[:2500]
    return f"""System: Du bist Faktencheck-Agent. Pruefe atomare Behauptungen gegen Quellkontext.

User:
Note-Titel: {note_title}
Note-Kontext: {compact_body}

Aufgabe: {stance}
Waehle pro Claim genau ein Label:
- supported_exact: woertliches Zitat oder fast woertlich, mindestens 80 Prozent lexikalische Ueberlappung
- supported_paraphrase: Sinn vorhanden, Wortwahl anders
- partially_supported: Teilaspekt belegt, Rest aus Synthese
- not_in_context: in den Top-5-Kontexten nicht belegt
- contradicted: Quelle widerspricht explizit

Evidence muss ein woertliches PDF-Zitat aus den gelieferten Kontexten sein, maximal 2 Saetze, sonst null.
best_page ist die passendste PDF-Seite als Integer oder null.

Few-Shot:
Claim: "Information Seeking umfasst aktive Suchhandlungen."
Kontext: "Information seeking is the purposive seeking for information..."
Output: {{"claim_idx": 0, "label": "supported_paraphrase", "evidence": "Information seeking is the purposive seeking for information", "justification": "Der Kontext belegt die aktive, zweckgerichtete Suche.", "best_page": 12}}
Claim: "Das Modell wurde 1999 empirisch repliziert."
Kontext: "The model was proposed in 1981."
Output: {{"claim_idx": 1, "label": "not_in_context", "evidence": null, "justification": "Eine Replikation 1999 wird nicht genannt.", "best_page": null}}
Claim: "Die Studie fand keine Unterschiede."
Kontext: "The study found significant differences between groups."
Output: {{"claim_idx": 2, "label": "contradicted", "evidence": "The study found significant differences between groups.", "justification": "Der Kontext sagt das Gegenteil.", "best_page": 4}}

Gib ausschliesslich ein JSON-Array zurueck. Keine Markdown-Fences, kein Begleittext.
"""


def _format_claims_for_prompt(items: list[RetrievedContext]) -> str:
    blocks: list[str] = []
    for item in items:
        ctx_lines = []
        for ctx in item.contexts:
            pages = ", ".join(str(p) for p in ctx["pages"]) or "unknown"
            ctx_lines.append(
                f"  Kontext {ctx['rank']} | Seiten {pages} | cosine {ctx['cosine']}:\n"
                f"  {ctx['text']}"
            )
        blocks.append(
            f"Claim idx={item.claim_idx}: {item.claim}\n"
            f"Top-Cosine: {item.top_cosine}\n"
            + "\n".join(ctx_lines)
        )
    return "\n\n".join(blocks)


def _split_for_prompt(note_title: str, note_body: str, items: list[RetrievedContext], *, variant: str) -> list[list[RetrievedContext]]:
    # CODEX-PATTERN: Single-call bleibt der Normalfall; nur bei langem Prompt wird deterministisch gesplittet,
    # damit der Judge nicht wegen Token-Limits ausfaellt und claim_idx trotzdem global stabil bleibt.
    batches: list[list[RetrievedContext]] = []
    current: list[RetrievedContext] = []
    for item in items:
        candidate = current + [item]
        prompt = _build_prompt(note_title, note_body, candidate, variant=variant)
        if current and len(prompt.split()) > MAX_PROMPT_WORDS:
            batches.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _build_prompt(note_title: str, note_body: str, items: list[RetrievedContext], *, variant: str = "primary") -> str:
    return _prompt_header(note_title, note_body, variant=variant) + "\nClaims und Kontexte:\n" + _format_claims_for_prompt(items)


def _json_array_from_text(text: str) -> list[Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("Claude output is not a JSON array")
    return data


def _repair_json_with_claude(raw_text: str, expected_indices: list[int], *, use_cache: bool) -> list[Any]:
    # CODEX-PATTERN: Reparatur ist ein eigener JSON-Normalisierungs-Call statt stiller Regex-Magie,
    # weil Schemafehler sonst schwer von echten Judge-Entscheidungen zu unterscheiden sind.
    prompt = f"""Extrahiere aus dem folgenden Text ein gueltiges JSON-Array.
Jedes Objekt braucht claim_idx, label, evidence, justification, best_page.
Erwartete claim_idx-Werte: {expected_indices}
Erlaubte Labels: {sorted(JUDGE_LABELS)}
Wenn ein Feld fehlt, setze evidence/best_page auf null und label auf "{PARSE_ERROR}".
Gib ausschliesslich JSON zurueck.

TEXT:
{raw_text}
"""
    repaired = base.call_claude_full(prompt, model=MODEL_OPUS, agent="eval_quality_v3_json_repair", use_cache=use_cache)
    return _json_array_from_text(repaired.text)


def _normalize_judge_rows(raw_rows: list[Any], items: list[RetrievedContext]) -> tuple[list[dict[str, Any]], list[str]]:
    expected = {item.claim_idx: item for item in items}
    rows_by_idx: dict[int, dict[str, Any]] = {}
    quality_flags: list[str] = []

    for raw in raw_rows:
        if not isinstance(raw, dict):
            quality_flags.append("judge_schema_row_invalid")
            continue
        try:
            idx = int(raw.get("claim_idx"))
        except (TypeError, ValueError):
            quality_flags.append("judge_schema_claim_idx_invalid")
            continue
        if idx not in expected:
            quality_flags.append("judge_schema_claim_idx_unexpected")
            continue
        if idx in rows_by_idx:
            quality_flags.append("duplicate_judge_response")
            continue
        label = str(raw.get("label") or "").strip()
        if label not in JUDGE_LABELS:
            label = PARSE_ERROR
            quality_flags.append("judge_schema_label_invalid")
        original_judge_label = label
        if expected[idx].top_cosine < RETRIEVAL_LOW_COSINE:
            # CODEX-PATTERN: Low-Cosine ist ein harter System-Override unabhaengig vom Judge-Label;
            # das Original bleibt fuer Drill-Down sichtbar, fliesst aber nicht in die Metrik.
            label = RETRIEVAL_UNCERTAIN
            quality_flags.append("retrieval_low_cosine")
        evidence = raw.get("evidence")
        if evidence is not None:
            evidence = str(evidence).strip() or None
        best_page = raw.get("best_page")
        try:
            best_page = int(best_page) if best_page is not None else None
        except (TypeError, ValueError):
            best_page = None
            quality_flags.append("judge_schema_best_page_invalid")
        rows_by_idx[idx] = {
            "claim_idx": idx,
            "label": label,
            "original_judge_label": original_judge_label,
            "evidence": evidence,
            "justification": str(raw.get("justification") or "").strip(),
            "best_page": best_page,
        }

    normalized: list[dict[str, Any]] = []
    for idx, item in expected.items():
        if idx not in rows_by_idx:
            quality_flags.append("judge_missing_claim")
            rows_by_idx[idx] = {
                "claim_idx": idx,
                "label": PARSE_ERROR,
                "original_judge_label": PARSE_ERROR,
                "evidence": None,
                "justification": "Judge lieferte kein Ergebnis fuer diesen Claim.",
                "best_page": item.best_page,
            }
        normalized.append(rows_by_idx[idx])
    normalized.sort(key=lambda row: row["claim_idx"])
    return normalized, sorted(set(quality_flags))


def _call_judge(note_title: str, note_body: str, items: list[RetrievedContext], *, variant: str, use_cache: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    meta = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_calls": 0, "quality_flags": []}
    for batch in _split_for_prompt(note_title, note_body, items, variant=variant):
        prompt = _build_prompt(note_title, note_body, batch, variant=variant)
        result = base.call_claude_full(prompt, model=MODEL_OPUS, agent=f"eval_quality_v3_{variant}", use_cache=use_cache)
        meta["calls"] += 1
        meta["input_tokens"] += result.input_tokens
        meta["output_tokens"] += result.output_tokens
        meta["cached_calls"] += 1 if result.cached else 0
        expected = [item.claim_idx for item in batch]
        try:
            raw_rows = _json_array_from_text(result.text)
        except Exception:
            try:
                raw_rows = _repair_json_with_claude(result.text, expected, use_cache=use_cache)
                meta["quality_flags"].append("judge_json_repaired")
            except Exception:
                # CODEX-PATTERN: Unheilbare JSON/Repair-Fehler werden claim-genau als parse_error
                # modelliert statt als Retrieval-Uncertainty in die Halluzinationsmetrik einzusickern.
                raw_rows = [
                    {
                        "claim_idx": idx,
                        "label": PARSE_ERROR,
                        "evidence": None,
                        "justification": "Judge-Ausgabe konnte nicht als JSON geparst werden.",
                        "best_page": None,
                    }
                    for idx in expected
                ]
                meta["quality_flags"].append("judge_json_parse_error")
        rows, flags = _normalize_judge_rows(raw_rows, batch)
        meta["quality_flags"].extend(flags)
        all_rows.extend(rows)
    all_rows.sort(key=lambda row: row["claim_idx"])
    meta["quality_flags"] = sorted(set(meta["quality_flags"]))
    return all_rows, meta


def _verification_text(pdf_path: Path) -> str:
    with fitz.open(str(pdf_path)) as pdf_doc:
        pages = [_extract_page_text(pdf_doc, page) for page in range(1, len(pdf_doc) + 1)]
    return " ".join(pages)


def _normalize_for_evidence(text: str) -> str:
    text = _normalize(text).lower()
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _verify_evidence(evidence: str | None, pdf_text: str) -> tuple[bool | None, float | None]:
    if not evidence:
        return None, None
    ev = _normalize_for_evidence(evidence)
    corpus = _normalize_for_evidence(pdf_text)
    if not ev or not corpus:
        return False, 0.0
    score = fuzz.token_set_ratio(ev, corpus) / 100.0
    return score >= 0.90, round(score, 3)


def _audit_indices(claim_scores: list[dict[str, Any]]) -> set[int]:
    triggers = {
        score["claim_idx"]
        for score in claim_scores
        if score["top_cosine"] < RETRIEVAL_LOW_COSINE or "evidence_unverified" in score["quality_flags"]
    }

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in claim_scores:
        by_label[score["label"]].append(score)

    # CODEX-PATTERN: Stratified audit ist stabil-hashbasiert statt random, damit wiederholte Eval-Laeufe
    # dieselben Claims pruefen und Dashboard-Diffs nicht durch Audit-Zufall rauschen.
    for rows in by_label.values():
        sample_n = max(1, math.ceil(len(rows) * 0.05))
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(f"{row['claim_idx']}|{row['claim']}".encode("utf-8")).hexdigest(),
        )
        triggers.update(row["claim_idx"] for row in ranked[:sample_n])
    return triggers


def _apply_audit_disagreements(
    claim_scores: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> None:
    audit_by_idx = {row["claim_idx"]: row for row in audit_rows}
    for score in claim_scores:
        audit = audit_by_idx.get(score["claim_idx"])
        if not audit:
            continue
        score["audit_label"] = audit["label"]
        score["audit_justification"] = audit["justification"]
        if audit["label"] != score["label"]:
            score["quality_flags"].append("judge_uneinig")
            score["label_original"] = score["label"]
            primary_rank = LABEL_STRICTNESS.get(score["label"])
            audit_rank = LABEL_STRICTNESS.get(audit["label"])
            # CODEX-PATTERN: Systemlabels sind keine Strenge-Stufe und werden nie durch Audit
            # ueberschrieben; Audit-Systemlabels markieren nur Unsicherheit der Zweitmeinung.
            if score["label"] in SYSTEM_LABELS:
                score["quality_flags"].append("audit_disagrees_with_system")
            elif audit["label"] in SYSTEM_LABELS:
                score["quality_flags"].append("audit_uncertain")
            elif audit_rank > primary_rank:
                score["label"] = audit["label"]
                score["quality_flags"].append("audit_overrode")
            else:
                score["quality_flags"].append("audit_disagrees_softer")


def _claim_scores_from_judge(
    claims: list[str],
    retrieved: list[RetrievedContext],
    judge_rows: list[dict[str, Any]],
    pdf_text: str,
) -> list[dict[str, Any]]:
    retrieved_by_idx = {item.claim_idx: item for item in retrieved}
    scores: list[dict[str, Any]] = []
    for row in judge_rows:
        item = retrieved_by_idx[row["claim_idx"]]
        verified, verification_score = _verify_evidence(row["evidence"], pdf_text)
        flags: list[str] = []
        label = row["label"]
        if verified is False:
            flags.append("evidence_unverified")
            if label == SUPPORTED_EXACT:
                label = NOT_IN_CONTEXT
                flags.append("evidence_fabricated")
            elif label == SUPPORTED_PARAPHRASE:
                label = PARTIALLY_SUPPORTED
                flags.append("evidence_fabricated")
        scores.append({
            "claim_idx": row["claim_idx"],
            "claim": claims[row["claim_idx"]],
            "label": label,
            "original_judge_label": row.get("original_judge_label"),
            "evidence": row["evidence"],
            "evidence_verified": verified,
            "evidence_verification_score": verification_score,
            "justification": row["justification"],
            "best_page": row["best_page"],
            "top_cosine": item.top_cosine,
            "best_chunk_idx": item.best_chunk_idx,
            "retrieved_contexts": item.contexts,
            "quality_flags": flags,
        })
    scores.sort(key=lambda score: score["claim_idx"])
    return scores


def _empty_result(note_path: Path, pdf_path: Path, pipeline_version: str, timestamp: str, error: str, total: int = 0) -> dict:
    return {
        "note": note_path.name,
        "pdf": pdf_path.name,
        "language": None,
        "version": pipeline_version,
        "eval_version": EVAL_VERSION,
        "timestamp": timestamp,
        "error": error,
        "claims_total": total,
        "claims_supported_exact": 0,
        "claims_supported_paraphrase": 0,
        "claims_partially_supported": 0,
        "claims_not_in_context": 0,
        "claims_contradicted": 0,
        "claims_retrieval_or_parse_uncertain": 0,
        "claims_parse_error": 0,
        "parse_error_count": 0,
        "valid_claims": None,
        "rate_valid": False,
        "confirmed_rate": -1.0,
        "partial_rate": 0.0,
        "retrieval_failure_rate": 0.0,
        "uncertain_rate": -1.0,
        "parse_error_rate": -1.0,
        "claim_support_rate": -1.0,
        "citation_verification_rate": -1.0,
        "claims_with_evidence": 0,
        "evidence_verified_count": 0,
        "anchors_total": total,
        "anchors_confirmed": 0,
        "anchors_hallucinated": 0,
        "hallucination_rate": -1.0,
        "coverage_rate": -1.0,
        "hallucination_ci_95": None,
        "pdf_chunks_total": 0,
        "claim_scores": [],
        "quality_flags": [],
        "llm_usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_calls": 0},
    }


def _aggregate(
    note_path: Path,
    pdf_path: Path,
    pipeline_version: str,
    timestamp: str,
    language_pair: str,
    chunks: list[Chunk],
    claim_scores: list[dict[str, Any]],
    llm_meta: dict[str, Any],
) -> dict:
    total = len(claim_scores)
    counts = Counter(score["label"] for score in claim_scores)
    # CODEX-PATTERN: parse_error ist ebenfalls nicht bewertbar und bleibt deshalb aus
    # dem Halluzinations-Nenner heraus, wird aber separat vom Retrieval-Unsicherheitswert berichtet.
    valid_claims = total - counts[RETRIEVAL_UNCERTAIN] - counts[PARSE_ERROR]
    parse_error_count = counts[PARSE_ERROR]
    confirmed = counts[SUPPORTED_EXACT] + counts[SUPPORTED_PARAPHRASE]
    hallucinated = counts[NOT_IN_CONTEXT] + counts[CONTRADICTED]
    with_evidence = sum(1 for score in claim_scores if score["evidence"])
    evidence_verified_count = sum(1 for score in claim_scores if score["evidence_verified"] is True)
    quality_flags = sorted({
        flag
        for score in claim_scores
        for flag in score["quality_flags"]
    } | set(llm_meta.get("quality_flags", [])))
    if parse_error_count:
        quality_flags = sorted(set(quality_flags) | {"parse_errors_present"})

    error = None
    # CODEX-PATTERN: Einzelne Parse-Errors entwerten nicht den ganzen Lauf; erst ab
    # 30 Prozent oder mindestens drei Fehlern wird das Ergebnis als unbrauchbar markiert.
    parse_error_limit = math.ceil(max(3, total * 0.30)) if total else 3
    if total and parse_error_count >= parse_error_limit:
        error = "too_many_parse_errors"
    elif valid_claims == 0:
        error = "no_valid_claims"

    rate_valid = error is None and valid_claims > 0
    # CODEX-PATTERN: Nicht messbare Raten bleiben Floats, damit bestehende Dashboards/Orchestrator
    # nicht an None brechen; rate_valid unterscheidet Sentinel von echten Messwerten.
    hallucination_rate = round(hallucinated / valid_claims, 3) if rate_valid else -1.0
    hallucination_ci_95 = wilson_ci(hallucinated, valid_claims) if rate_valid else None
    confirmed_rate = round(confirmed / valid_claims, 3) if rate_valid else -1.0
    claim_support_rate = round((confirmed + counts[PARTIALLY_SUPPORTED]) / valid_claims, 3) if rate_valid else -1.0

    result = {
        "note": note_path.name,
        "pdf": pdf_path.name,
        "language": language_pair,
        "version": pipeline_version,
        "eval_version": EVAL_VERSION,
        "timestamp": timestamp,
        "claims_total": total,
        "claims_supported_exact": counts[SUPPORTED_EXACT],
        "claims_supported_paraphrase": counts[SUPPORTED_PARAPHRASE],
        "claims_partially_supported": counts[PARTIALLY_SUPPORTED],
        "claims_not_in_context": counts[NOT_IN_CONTEXT],
        "claims_contradicted": counts[CONTRADICTED],
        "claims_retrieval_or_parse_uncertain": counts[RETRIEVAL_UNCERTAIN],
        "claims_parse_error": parse_error_count,
        "parse_error_count": parse_error_count,
        "valid_claims": valid_claims,
        "rate_valid": rate_valid,
        "confirmed_rate": confirmed_rate,
        "partial_rate": round(counts[PARTIALLY_SUPPORTED] / total, 3) if total else 0.0,
        "retrieval_failure_rate": round(counts[RETRIEVAL_UNCERTAIN] / total, 3) if total else 0.0,
        "uncertain_rate": round(counts[RETRIEVAL_UNCERTAIN] / total, 3) if total else -1.0,
        "parse_error_rate": round(parse_error_count / total, 3) if total else -1.0,
        "claim_support_rate": claim_support_rate,
        "citation_verification_rate": round(evidence_verified_count / with_evidence, 3) if with_evidence else -1.0,
        "claims_with_evidence": with_evidence,
        "evidence_verified_count": evidence_verified_count,
        "anchors_total": total,
        "anchors_confirmed": confirmed,
        "anchors_hallucinated": hallucinated,
        "hallucination_rate": hallucination_rate,
        "coverage_rate": confirmed_rate,
        "hallucination_ci_95": hallucination_ci_95,
        "pdf_chunks_total": len(chunks),
        "claim_scores": claim_scores,
        "quality_flags": quality_flags,
        "llm_usage": {
            "calls": llm_meta.get("calls", 0),
            "input_tokens": llm_meta.get("input_tokens", 0),
            "output_tokens": llm_meta.get("output_tokens", 0),
            "cached_calls": llm_meta.get("cached_calls", 0),
        },
    }
    if error:
        result["error"] = error
    return result


def eval_note(note_path: Path | str, pdf_path: Path | str, pipeline_version: str = AGENT_VERSION,
              no_cache: bool = False) -> dict:
    """Evaluiert eine Note gegen ihre Quell-PDF und gibt v3-Metriken zurueck."""
    note_path = Path(note_path)
    pdf_path = Path(pdf_path)
    timestamp = datetime.now().isoformat()
    use_cache = not no_cache

    if not note_path.exists():
        return _empty_result(note_path, pdf_path, pipeline_version, timestamp, "note_not_found")
    if not pdf_path.exists():
        return _empty_result(note_path, pdf_path, pipeline_version, timestamp, "pdf_not_found")

    note_body = _read_note_body(note_path)
    claims = extract_claims(note_path)
    if not claims:
        return _empty_result(note_path, pdf_path, pipeline_version, timestamp, "no_claims_found")

    chunks = build_chunks(pdf_path)
    if not chunks:
        result = _empty_result(note_path, pdf_path, pipeline_version, timestamp, "pdf_not_parseable", total=len(claims))
        result["claim_scores"] = []
        return result

    retrieved = _retrieve_claim_contexts(claims, chunks)
    pdf_text = _verification_text(pdf_path)
    language_pair = _detect_language_pair(note_body, chunks[0].text if chunks else "")
    note_title = _note_title(note_path, note_body)

    judge_rows, llm_meta = _call_judge(note_title, note_body, retrieved, variant="primary", use_cache=use_cache)
    claim_scores = _claim_scores_from_judge(claims, retrieved, judge_rows, pdf_text)

    audit_indices = _audit_indices(claim_scores)
    if audit_indices:
        # CODEX-PATTERN: Der Re-Run verwendet denselben Judge-Pfad mit anderer Prompt-Variante,
        # damit Audit und Primaerbewertung keine parallelen Parser/Schema-Implementierungen driften lassen.
        audit_items = [item for item in retrieved if item.claim_idx in audit_indices]
        audit_rows, audit_meta = _call_judge(note_title, note_body, audit_items, variant="audit", use_cache=use_cache)
        _apply_audit_disagreements(claim_scores, audit_rows)
        llm_meta["calls"] += audit_meta.get("calls", 0)
        llm_meta["input_tokens"] += audit_meta.get("input_tokens", 0)
        llm_meta["output_tokens"] += audit_meta.get("output_tokens", 0)
        llm_meta["cached_calls"] += audit_meta.get("cached_calls", 0)
        llm_meta["quality_flags"] = sorted(set(llm_meta.get("quality_flags", [])) | set(audit_meta.get("quality_flags", [])))

    return _aggregate(note_path, pdf_path, pipeline_version, timestamp, language_pair, chunks, claim_scores, llm_meta)


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
    hallucination = result["hallucination_rate"]
    hallucination_text = "n/a" if not result.get("rate_valid", hallucination >= 0) else f"{hallucination:.1%}"
    print(
        f"[eval_quality_v3] {result['note']}: "
        f"{result['anchors_confirmed']}/{result['claims_total']} confirmed, "
        f"{result['claims_partially_supported']} partial, "
        f"{result['anchors_hallucinated']} hallucinated, "
        f"{result['claims_retrieval_or_parse_uncertain']} retrieval_uncertain, "
        f"hallucination={hallucination_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge Qualitaetsmessung")
    parser.add_argument("--note", help="Pfad zur Note-Datei (.md)")
    parser.add_argument("--pdf", help="Pfad zur Quell-PDF")
    parser.add_argument("--version", default=AGENT_VERSION, help="Pipeline-Version")
    parser.add_argument("--save", action="store_true", help="Ergebnis in quality_history.jsonl speichern")
    parser.add_argument("--no-cache", action="store_true", help="Claude-Cache fuer diesen Lauf umgehen")
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
