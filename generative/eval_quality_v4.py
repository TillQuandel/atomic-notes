#!/usr/bin/env python3
"""LLM-as-Judge Qualitaetsmessung fuer Atomic-Agent Notes.

v4 bewertet atomare Claims gegen Top-K-PDF-Kontexte mit Claude als Judge,
verifiziert angegebene Zitate programmatisch und aggregiert Retrieval-Failures
separat von Halluzinationen. Die finale Decision-Logic lebt im modularen
decision_engine-Paket.
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
from generative.config import AGENT_VERSION, CACHE_DIR, EVAL_ADAPTIVE_K_HIGH, EVAL_ADAPTIVE_K_MID, MODEL_OPUS, MODEL_CONFIG, QUALITY_HISTORY
from decision_engine import ClaimDecision, ClaimInput, DEFAULT_CONFIG, Label, determine_decision
from decision_engine.aggregation import aggregate as aggregate_decisions
from decision_engine.models import QualityFlag
from generative.eval_quality import _extract_page_text, _normalize, wilson_ci
from generative.eval_quality_v2 import TOP_K, Chunk, _detect_language_pair, _expand_context, _read_note_body
from generative.eval_quality_v2 import build_chunks, extract_claims
from generative.pipeline.embeddings import _model, cosine

_QUALITY_HISTORY = QUALITY_HISTORY  # SSoT: config.QUALITY_HISTORY; Alias für bestehende Importer (run.py, adversarial.py)
EVAL_VERSION = "4.1"
_ENGINE_FLAG_VALUES = {flag.value for flag in QualityFlag}

SUPPORTED_EXACT = Label.SUPPORTED_EXACT.value
SUPPORTED_PARAPHRASE = Label.SUPPORTED_PARAPHRASE.value
PARTIALLY_SUPPORTED = Label.PARTIALLY_SUPPORTED.value
NOT_IN_CONTEXT = Label.NOT_IN_CONTEXT.value
CONTRADICTED = Label.CONTRADICTED.value
RETRIEVAL_UNCERTAIN = Label.RETRIEVAL_UNCERTAIN.value
PARSE_ERROR = Label.PARSE_ERROR.value
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
# CLAUDE-PATTERN: Threshold ist Single Source of Truth in decision_engine.RulesConfig.
# Lokale Konstante bezieht sich davon, damit kein Drift zwischen Eval-Audit-Trigger und Pipeline.
RETRIEVAL_LOW_COSINE = DEFAULT_CONFIG.retrieval_low_cosine_threshold
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
        all_ranked = sorted(
            ((idx, cosine(chunk_embs[idx], claim_emb)) for idx in range(len(chunks))),
            key=lambda item: item[1],
            reverse=True,
        )
        # Margin-basiertes adaptives TOP_K (Gemini-Review 2026-05-22):
        # Dichte Score-Cluster (≤0.05 Abstand) → k groß halten (Retriever unsicher).
        # Starker Score-Abfall → k=2 aggressiv (eindeutiger Treffer, Tokens sparen).
        # Löst False-Negatives bei paraphrase-multilingual-MiniLM wo Scores dicht clustern.
        top_score = all_ranked[0][1] if all_ranked else 0.0
        margin_threshold = top_score - 0.05
        adaptive_k = 2  # Absolutes Minimum
        for _idx, (_, _score) in enumerate(all_ranked):
            if _score >= margin_threshold:
                adaptive_k = max(adaptive_k, _idx + 1)
            else:
                break
        adaptive_k = min(adaptive_k, TOP_K)  # Obergrenze bei 5
        ranked = all_ranked[:adaptive_k]
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


def _prompt_header(note_title: str, *, variant: str) -> str:
    if variant == "audit":
        stance = "Pruefe skeptisch, ob der Claim wirklich aus dem Kontext folgt. Bevorzuge strenge Labels bei Retrieval-Problemen."
    else:
        stance = "Pruefe fair, ob der Claim im Kontext exakt, sinngemaess, teilweise, nicht oder widerspruechlich belegt ist."
    # note_body absichtlich weggelassen — Gemini-Review 2026-05-18: Note-Kontext erzeugt
    # Confirmation Bias (Judge bestaetigt Claims anhand des Note-Textes statt strikt gegen
    # PDF-Chunks zu pruefen). Nur Titel bleibt fuer Disambiguierung.
    return f"""System: Du bist Faktencheck-Agent. Pruefe atomare Behauptungen gegen Quellkontext.

User:
Note-Titel: {note_title}

Aufgabe: {stance}

Kontext-Pool-Instruktion: Die Quellen sind im Abschnitt "Kontext-Pool" abgelegt.
Jeder Claim listet Kontext-IDs (z.B. K1, K2) — schlage diese IDs im Pool nach um den Claim zu pruefen.

Waehle pro Claim genau ein Label:
- supported_exact: woertliches Zitat oder fast woertlich, mindestens 80 Prozent lexikalische Ueberlappung
- supported_paraphrase: Sinn vorhanden, Wortwahl anders
- partially_supported: Teilaspekt belegt, Rest aus Synthese
- not_in_context: in den gelieferten Kontexten nicht belegt
- contradicted: Quelle widerspricht explizit

Evidence-Pflicht: Gib fuer "evidence" zwingend den woertlichen Textauszug aus dem Kontext-Pool zurueck
(maximal 2 Saetze), NIEMALS nur eine ID wie "K1". Wenn kein Beleg: null.
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


def _build_context_pool(
    items: list[RetrievedContext],
) -> tuple[dict[int, str], str]:
    """Baut deduplizierten Kontext-Pool aus allen Claims.

    Jeder unique chunk_idx bekommt eine Pool-ID (K1, K2, ...).
    Chunks die mehrere Claims teilen erscheinen nur einmal im Prompt —
    spart 40-60% Input-Tokens bei hohem Chunk-Overlap (Gemini-Review 2026-05-18).

    Returns:
        chunk_to_kid: {chunk_idx: "K1", ...}
        pool_text: formatierter Pool-Abschnitt
    """
    seen: dict[int, str] = {}
    pool_entries: list[str] = []
    for item in items:
        for ctx in item.contexts:
            cid = ctx["chunk_idx"]
            if cid not in seen:
                kid = f"K{len(seen) + 1}"
                seen[cid] = kid
                pages = ", ".join(str(p) for p in ctx["pages"]) or "unbekannt"
                pool_entries.append(f"[{kid}] Seiten {pages}:\n{ctx['text']}")
    return seen, "\n\n".join(pool_entries)


def _format_claims_for_prompt(
    items: list[RetrievedContext],
    chunk_to_kid: dict[int, str],
) -> str:
    """Formatiert Claims mit Pool-Referenzen (K1, K2, ...) statt Inline-Kontext."""
    blocks: list[str] = []
    for item in items:
        # dict.fromkeys: Reihenfolge erhalten + Duplikate entfernen falls chunk_idx mehrfach vorkommt
        kids = list(dict.fromkeys(
            chunk_to_kid[ctx["chunk_idx"]]
            for ctx in item.contexts
            if ctx["chunk_idx"] in chunk_to_kid
        ))
        blocks.append(
            f"idx={item.claim_idx} | top-cosine={item.top_cosine} | Kontexte: {', '.join(kids)}\n"
            f"Claim: \"{item.claim}\""
        )
    return "\n\n".join(blocks)


def _split_for_prompt(note_title: str, items: list[RetrievedContext], *, variant: str) -> list[list[RetrievedContext]]:
    # CODEX-PATTERN: Single-call bleibt der Normalfall; nur bei langem Prompt wird deterministisch gesplittet,
    # damit der Judge nicht wegen Token-Limits ausfaellt und claim_idx trotzdem global stabil bleibt.
    batches: list[list[RetrievedContext]] = []
    current: list[RetrievedContext] = []
    for item in items:
        candidate = current + [item]
        prompt = _build_prompt(note_title, candidate, variant=variant)
        if current and len(prompt.split()) > MAX_PROMPT_WORDS:
            batches.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _build_prompt(note_title: str, items: list[RetrievedContext], *, variant: str = "primary") -> str:
    chunk_to_kid, pool_text = _build_context_pool(items)
    claims_text = _format_claims_for_prompt(items, chunk_to_kid)
    return (
        _prompt_header(note_title, variant=variant)
        + "\n## Kontext-Pool\n"
        + pool_text
        + "\n\n## Claims\n"
        + claims_text
    )


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
    repaired = base.call_llm_full(prompt, model=MODEL_OPUS, agent="eval_quality_v3_json_repair", use_cache=use_cache)
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


def _call_judge(note_title: str, items: list[RetrievedContext], *, variant: str, use_cache: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    meta = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_calls": 0, "quality_flags": []}
    for batch in _split_for_prompt(note_title, items, variant=variant):
        prompt = _build_prompt(note_title, batch, variant=variant)
        result = base.call_llm_full(prompt, model=MODEL_OPUS, agent=f"eval_quality_v3_{variant}", use_cache=use_cache)
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


def _audit_indices(claim_scores: list[dict[str, Any]], note_title: str = "") -> set[int]:
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
    # Rate 0.05 → 0.02 (Gemini-Review 2026-05-18): bei 10-15 Claims/Note ergibt
    # ceil(0.05*N) meist 1-2 pro Klasse, ceil(0.02*N) auch fast immer 1 — kein
    # messbarer Quality-Unterschied, spart ~10% Audit-Calls.
    for rows in by_label.values():
        sample_n = max(1, math.ceil(len(rows) * 0.02))
        ranked = sorted(
            rows,
            # Hash ueber note_title|claim — stabil auch bei Index-Verschiebung,
            # eindeutig auch wenn gleicher Claim-Text in verschiedenen Notes vorkommt.
            key=lambda row: hashlib.sha256(
                f"{note_title}|{row['claim']}".encode("utf-8")
            ).hexdigest(),
        )
        triggers.update(row["claim_idx"] for row in ranked[:sample_n])
    return triggers


def _claim_scores_from_judge(
    claims: list[str],
    retrieved: list[RetrievedContext],
    judge_rows: list[dict[str, Any]],
    pdf_text: str,
    audit_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    retrieved_by_idx = {item.claim_idx: item for item in retrieved}
    audit_by_idx = {row["claim_idx"]: row for row in audit_rows or []}
    scores: list[dict[str, Any]] = []
    for row in judge_rows:
        item = retrieved_by_idx[row["claim_idx"]]
        verified, verification_score = _verify_evidence(row["evidence"], pdf_text)
        audit = audit_by_idx.get(row["claim_idx"])
        audit_label = Label(audit["label"]) if audit else None
        # CODEX-PATTERN: v4 converts Judge rows into a domain-agnostic ClaimInput and lets
        # decision_engine own low-cosine, evidence downgrade, and audit precedence.
        decision = determine_decision(ClaimInput(
            primary_label=Label(row["label"]),
            audit_label=audit_label,
            cosine=item.top_cosine,
            evidence_verified=verified,
            parse_failed=row["label"] == PARSE_ERROR,
        ))
        flags = [flag.value for flag in decision.flags]
        scores.append({
            "claim_idx": row["claim_idx"],
            "claim": claims[row["claim_idx"]],
            "label": decision.label.value,
            "decision_source": decision.source,
            "original_judge_label": row.get("original_judge_label"),
            "label_original": row["label"] if decision.label.value != row["label"] else None,
            "audit_label": audit["label"] if audit else None,
            "audit_justification": audit["justification"] if audit else None,
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
    decisions = [
        ClaimDecision(
            Label(score["label"]),
            frozenset(QualityFlag(flag) for flag in score["quality_flags"] if flag in _ENGINE_FLAG_VALUES),
            score.get("decision_source", "primary"),
        )
        for score in claim_scores
    ]
    engine_metrics = aggregate_decisions(decisions)
    # CODEX-PATTERN: Aggregationsraten und Label-Zaehler kommen aus decision_engine; v4 reichert nur
    # Atomic-Agent-spezifische Felder wie Wilson-CI, PDF-Chunk-Zahl und LLM-Usage an.
    valid_claims = int(engine_metrics["valid_claims"])
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

    # CLAUDE-PATTERN: v4.1-Aggregation hat per-Metric validity via nested metrics-Dict;
    # rate_valid für die Halluzinationsrate aus metrics["hallucination_rate"]["valid"] ziehen.
    hall_metric = engine_metrics["metrics"]["hallucination_rate"]
    rate_valid = error is None and valid_claims > 0 and bool(hall_metric["valid"])
    hallucination_rate = float(hall_metric["value"]) if rate_valid else -1.0
    hallucination_ci_95 = wilson_ci(hallucinated, valid_claims) if rate_valid else None
    conf_metric = engine_metrics["metrics"]["confirmed_rate"]
    confirmed_rate = float(conf_metric["value"]) if conf_metric["valid"] and error is None else -1.0
    support_metric = engine_metrics["metrics"]["claim_support_rate"]
    claim_support_rate = float(support_metric["value"]) if support_metric["valid"] and error is None else -1.0

    result = {
        "note": note_path.name,
        "pdf": pdf_path.name,
        "language": language_pair,
        "version": pipeline_version,
        "eval_version": EVAL_VERSION,
        "timestamp": timestamp,
        "claims_total": engine_metrics["claims_total"],
        "claims_supported_exact": engine_metrics["claims_supported_exact"],
        "claims_supported_paraphrase": engine_metrics["claims_supported_paraphrase"],
        "claims_partially_supported": engine_metrics["claims_partially_supported"],
        "claims_not_in_context": engine_metrics["claims_not_in_context"],
        "claims_contradicted": engine_metrics["claims_contradicted"],
        "claims_retrieval_or_parse_uncertain": engine_metrics["claims_retrieval_or_parse_uncertain"],
        "claims_parse_error": engine_metrics["claims_parse_error"],
        "parse_error_count": parse_error_count,
        "valid_claims": valid_claims,
        "rate_valid": rate_valid,
        "confirmed_rate": confirmed_rate,
        "partial_rate": engine_metrics["partial_rate"],
        "retrieval_failure_rate": engine_metrics["retrieval_failure_rate"],
        "uncertain_rate": engine_metrics["uncertain_rate"],
        "parse_error_rate": engine_metrics["parse_error_rate"],
        "claim_support_rate": claim_support_rate,
        "citation_verification_rate": round(evidence_verified_count / with_evidence, 3) if with_evidence else -1.0,
        "claims_with_evidence": with_evidence,
        "evidence_verified_count": evidence_verified_count,
        "anchors_total": engine_metrics["anchors_total"],
        "anchors_confirmed": engine_metrics["anchors_confirmed"],
        "anchors_hallucinated": engine_metrics["anchors_hallucinated"],
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
        "model_config": MODEL_CONFIG,
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

    judge_rows, llm_meta = _call_judge(note_title, retrieved, variant="primary", use_cache=use_cache)
    claim_scores = _claim_scores_from_judge(claims, retrieved, judge_rows, pdf_text)

    audit_indices = _audit_indices(claim_scores, note_title)
    if audit_indices:
        # CODEX-PATTERN: Der Re-Run verwendet denselben Judge-Pfad mit anderer Prompt-Variante,
        # damit Audit und Primaerbewertung keine parallelen Parser/Schema-Implementierungen driften lassen.
        audit_items = [item for item in retrieved if item.claim_idx in audit_indices]
        audit_rows, audit_meta = _call_judge(note_title, audit_items, variant="audit", use_cache=use_cache)
        claim_scores = _claim_scores_from_judge(claims, retrieved, judge_rows, pdf_text, audit_rows)
        llm_meta["calls"] += audit_meta.get("calls", 0)
        llm_meta["input_tokens"] += audit_meta.get("input_tokens", 0)
        llm_meta["output_tokens"] += audit_meta.get("output_tokens", 0)
        llm_meta["cached_calls"] += audit_meta.get("cached_calls", 0)
        llm_meta["quality_flags"] = sorted(set(llm_meta.get("quality_flags", [])) | set(audit_meta.get("quality_flags", [])))

    return _aggregate(note_path, pdf_path, pipeline_version, timestamp, language_pair, chunks, claim_scores, llm_meta)


def save_result(result: dict) -> None:
    """Appended Ergebnis an quality_history.jsonl und atomic_analytics.db."""
    _QUALITY_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with _QUALITY_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # DB: note_eval persistieren
    try:
        from generative import db as _db
        from generative.agents.base import _RUN_ID as _run_id
        note_name = result.get("note", "")
        eval_id = f"{_run_id}__{note_name}"
        with _db.get_db() as _conn:
            _db.insert_eval(_conn, {
                "eval_id":           eval_id,
                "run_id":            _run_id,
                "note_path":         note_name,
                "acceptance_status": None,  # wird vom orchestrator gesetzt
                "hallucination_rate":result.get("hallucination_rate"),
                "anchors_total":     result.get("anchors_total"),
                "anchors_hallucinated": result.get("anchors_hallucinated"),
                "coverage_factual":  result.get("coverage_factual"),
                "coverage_rate":     result.get("coverage_rate"),
                "tokens_total":      result.get("tokens_total"),
                "tokens_input":      result.get("tokens_input"),
                "tokens_output":     result.get("tokens_output"),
                "tokens_cache_read": result.get("tokens_cache_read"),
                "wall_time_s":       result.get("wall_time_s"),
                "pipeline_version":  result.get("version"),
                "pdf":               result.get("pdf"),
                "language":          result.get("language"),
                "eval_version":      result.get("eval_version", EVAL_VERSION),
                "timestamp":         result.get("timestamp"),
            })
    except Exception as _db_err:
        import sys as _sys
        print(f"  [warn] DB-Write fehlgeschlagen: {_db_err}", file=_sys.stderr)

    print(f"  -> gespeichert: {_QUALITY_HISTORY}")


def print_summary(result: dict) -> None:
    if "error" in result:
        print(f"[ERROR] {result['note']}: {result['error']}")
        return
    hallucination = result["hallucination_rate"]
    hallucination_text = "n/a" if not result.get("rate_valid", hallucination >= 0) else f"{hallucination:.1%}"
    print(
        f"[eval_quality_v4] {result['note']}: "
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
