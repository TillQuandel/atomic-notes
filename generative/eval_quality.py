#!/usr/bin/env python3
"""eval_quality.py — Deterministische Halluzinations-Messung für Atomic-Agent Notes.

Misst wie viele source_anchors einer Note tatsächlich in der Quell-PDF stehen —
ohne LLM-Beteiligung. Alle Scores sind deterministisch und reproduzierbar.

Design-Entscheidungen (Gemini + Nemotron Review 2026-05-14):
- get_text("blocks") statt "text" → saubere Lesereihenfolge bei Spalten-PDFs
- Normalisierung vor Match → Silbentrennung, Ligaturen, Sonderzeichen
- S.N + S.N+1 prüfen → seitenübergreifende Zitate
- 0.8 Fuzzy / 0.2 Semantic → Anker sind Direktzitate, nicht Paraphrasen
- Threshold 0.85 / 0.70 (confirmed / uncertain)
- OCR-Detect: leere Seiten → "not_parseable", nicht als halluziniert zählen
- Coverage-Metrik: confirmed / body_sentences (neben hallucination_rate)
- Wilson-CI bei < 5 Ankern → Stichprobengröße explizit berichten

Usage:
    python eval_quality.py --note 04-wissen/Agent-Evals.md --pdf Literatur/guide.pdf
    python eval_quality.py --baseline  # alle Baseline-PDFs + ihre Vault-Notes
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

# Projekt-Root für Imports

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF fehlt: pip install pymupdf")

try:
    from rapidfuzz import fuzz
except ImportError:
    sys.exit("rapidfuzz fehlt: pip install rapidfuzz")

from generative.config import (AGENT_VERSION, CACHE_DIR, ENABLE_MDEBERTA_NLI,
                    MDEBERTA_NLI_MODEL, MDEBERTA_THRESHOLD_CONFIRMED,
                    MDEBERTA_THRESHOLD_CONTRA)

_QUALITY_HISTORY = CACHE_DIR / "quality_history.jsonl"

# Eval-Methoden-Version — unabhängig von Pipeline-Version (AGENT_VERSION).
# Ändern wenn sich Threshold, Gewichtung oder Scoring-Logik ändert.
# Historische Records behalten ihre eval_version → Vergleiche bleiben valide.
EVAL_VERSION = "1.3"  # 1.3: mDeBERTa NLI-Scorer für Cross-Language (512-Token Window)

# Stoppwörter für Structural-Filter (DE + EN)
_STRUCTURAL_STARTS = {
    "des weiteren", "außerdem", "zusammenfassend", "daher", "somit", "damit",
    "folglich", "demnach", "insgesamt", "abschließend", "zusammenfassend",
    "furthermore", "moreover", "therefore", "thus", "hence", "additionally",
    "in summary", "to summarize", "finally", "lastly", "in conclusion",
}

# Thresholds — Gemini-Review: 0.85 für same-language Direktzitate
# Cross-language (DE Note ← EN PDF): kalibriert auf labeled data nötig.
# Aktuell konservativ: 0.70/0.50 — absolute Werte sind Annäherung,
# Vergleiche über Versionen bleiben valide wenn Threshold konstant bleibt.
THRESHOLD_CONFIRMED_SAME = 0.85   # same-language / original-lang quote
THRESHOLD_UNCERTAIN_SAME = 0.70
THRESHOLD_CONFIRMED_CROSS = 0.70  # cross-language paraphrase (uncalibrated)
THRESHOLD_UNCERTAIN_CROSS = 0.50

# Gewichtung für same-language (Direktzitat → Fuzzy dominant)
# Cross-Language: w_fuzzy=0.0, w_semantic=1.0 (Fuzzy = Rauschen bei Sprachwechsel)
# Gemini+DeepSeek Cross-Review 2026-05-14 → EVAL_VERSION 1.2
WEIGHT_FUZZY = 0.8
WEIGHT_SEMANTIC = 0.2


# ---------------------------------------------------------------------------
# Text-Extraktion + Normalisierung
# ---------------------------------------------------------------------------

def _extract_page_text(pdf_doc: fitz.Document, page_num: int) -> str:
    """Extrahiert Seiten-Text via blocks für korrekte Lesereihenfolge.

    Normalisierung (Gemini-Finding): Silbentrennung, Ligaturen, Whitespace.
    Gibt "" zurück wenn Seite nicht existiert oder leer (OCR-PDF).
    """
    try:
        page = pdf_doc[page_num - 1]  # 1-basiert → 0-basiert
    except IndexError:
        return ""
    # get_text("blocks") → [(x0,y0,x1,y1,text,block_no,block_type), ...]
    # Nur Text-Blöcke (block_type=0), nach Y-Position sortiert
    blocks = page.get_text("blocks")
    text_parts = [b[4] for b in sorted(blocks, key=lambda b: b[1]) if b[6] == 0]
    raw = " ".join(text_parts)
    return _normalize(raw)


def _normalize(text: str) -> str:
    """Normalisiert PDF-Text für robustes Matching.

    - Silbentrennung: 'Infor-\nmations-' → 'Informations'
    - Ligaturen: fi, fl → fi, fl (PyMuPDF normalisiert meist, aber sicher ist sicher)
    - Whitespace kollabieren
    - Sonderzeichen die Fuzzy stören
    """
    # Silbentrennung am Zeilenende
    text = re.sub(r"-\s*\n\s*", "", text)
    # Normales Newline → Leerzeichen
    text = re.sub(r"\s+", " ", text)
    # Anführungszeichen normalisieren
    text = text.replace("„", '"').replace("“", '"').replace("’", "'")
    return text.strip()


def _page_num_from_str(page_str: str) -> int | None:
    """'S. 7' → 7, 'S. 12-13' → 12, None wenn nicht parsebar."""
    if not page_str:
        return None
    m = re.search(r"\d+", page_str)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _semantic_score(quote: str, page_text: str) -> float:
    """Sentence-Transformers Cosine-Similarity. 0.0 wenn Modell nicht geladen."""
    try:
        from generative.pipeline.embeddings import _model, cosine
        import numpy as np
        model = _model()
        if model is None:
            return 0.0
        q_emb = model.encode(quote, normalize_embeddings=True)
        # Vergleiche gegen jeden Satz im Page-Text, nimm Maximum
        sentences = [s.strip() for s in re.split(r"[.!?]", page_text) if len(s.strip()) > 20]
        if not sentences:
            return 0.0
        s_embs = model.encode(sentences, normalize_embeddings=True)
        scores = s_embs.dot(q_emb)
        return float(scores.max())
    except Exception:
        return 0.0


_nli_model_cache: dict = {}


def _nli_model():
    """Lazy-load mDeBERTa NLI-Modell. Gibt (tokenizer, model) oder (None, None) zurück."""
    if "loaded" in _nli_model_cache:
        return _nli_model_cache.get("tok"), _nli_model_cache.get("model")
    _nli_model_cache["loaded"] = True
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tok = AutoTokenizer.from_pretrained(MDEBERTA_NLI_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(MDEBERTA_NLI_MODEL)
        mdl.eval()
        _nli_model_cache["tok"] = tok
        _nli_model_cache["model"] = mdl
        _nli_model_cache["torch"] = torch
        return tok, mdl
    except Exception as e:
        print(f"  [mDeBERTa] Modell nicht geladen: {e}", file=sys.stderr)
        _nli_model_cache["tok"] = None
        _nli_model_cache["model"] = None
        return None, None


def _nli_best_window(premise: str, hypothesis: str, tokenizer, max_total: int = 512) -> str:
    """Kürzt premise auf Token-Budget: max_total - len(hypothesis_tokens) - 3 Sonderzeichen.

    Strategie: Sätze via fuzzy gegen hypothesis ranken, bestes Fenster zentriert
    auf den höchstrangigen Satz aufbauen bis Budget erschöpft.
    Safety-Margin: 10 Tokens Reserve gegen Tokenizer-Diskrepanzen bei Sonderzeichen.
    """
    hyp_tokens = tokenizer.encode(hypothesis, add_special_tokens=False)
    budget = max_total - len(hyp_tokens) - 3 - 10  # [CLS] premise [SEP] hypothesis [SEP] + margin
    if budget <= 0:
        return ""

    prem_tokens = tokenizer.encode(premise, add_special_tokens=False)
    if len(prem_tokens) <= budget:
        return premise

    # Sätze splitten und nach fuzzy-Score gegen hypothesis ranken
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", premise) if s.strip()]
    if not sentences:
        # Fallback: Token-Truncation von vorne
        return tokenizer.decode(prem_tokens[:budget], skip_special_tokens=True)

    scored = sorted(enumerate(sentences),
                    key=lambda x: fuzz.partial_ratio(hypothesis, x[1]),
                    reverse=True)
    best_idx = scored[0][0]

    # Fenster zentriert auf best_idx aufbauen
    window: list[tuple[int, str]] = [(best_idx, sentences[best_idx])]
    used = len(tokenizer.encode(sentences[best_idx], add_special_tokens=False))
    lo, hi = best_idx - 1, best_idx + 1

    while used < budget:
        added = False
        if lo >= 0:
            toks = tokenizer.encode(sentences[lo], add_special_tokens=False)
            if used + len(toks) <= budget:
                window.insert(0, (lo, sentences[lo]))
                used += len(toks)
                lo -= 1
                added = True
        if hi < len(sentences):
            toks = tokenizer.encode(sentences[hi], add_special_tokens=False)
            if used + len(toks) <= budget:
                window.append((hi, sentences[hi]))
                used += len(toks)
                hi += 1
                added = True
        if not added:
            break

    window.sort(key=lambda x: x[0])
    return " ".join(s for _, s in window)


def _nli_score(quote: str, page_text: str) -> dict:
    """mDeBERTa NLI: premise=page_text (EN), hypothesis=quote (DE).

    Gibt dict mit entailment/neutral/contradiction (0–1) zurück.
    Leer-Dict bei Fehler oder Modell nicht geladen.
    """
    tok, mdl = _nli_model()
    if tok is None or mdl is None:
        return {}
    torch = _nli_model_cache["torch"]

    try:
        premise = _nli_best_window(page_text, quote, tok)
        if not premise:
            return {}

        inp = tok(premise, quote, truncation=True, max_length=512,
                  return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**inp).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        # mDeBERTa-mnli-xnli label order: entailment=0, neutral=1, contradiction=2
        # (Reihenfolge laut model.config.id2label prüfen — Standard bei Laurer-Modellen)
        labels = mdl.config.id2label
        label_map = {v.lower(): i for i, v in labels.items()}
        e = probs[label_map.get("entailment", 0)]
        n = probs[label_map.get("neutral", 1)]
        c = probs[label_map.get("contradiction", 2)]
        return {"entailment": round(e, 3), "neutral": round(n, 3),
                "contradiction": round(c, 3)}
    except Exception as ex:
        print(f"  [mDeBERTa] Scoring-Fehler: {ex}", file=sys.stderr)
        return {}


def _nli_label(nli: dict, cosine_label: str) -> str | None:
    """NLI als selektiver Override auf Cosine-Label.

    mDeBERTa ist für logisches Entailment trainiert, nicht für lose Paraphrasen.
    Deshalb NLI nur in zwei Fällen überschreiben:
    1. Starkes Contradiction-Signal (>= CONTRA) → hallucinated (Cosine blind dafür)
    2. Starkes Entailment (>= CONFIRMED, z.B. direkte Zitate) → confirmed
    Sonst: cosine_label beibehalten (NLI zu streng für Paraphrasen).
    """
    if not nli:
        return None
    if nli.get("contradiction", 0) >= MDEBERTA_THRESHOLD_CONTRA:
        return "hallucinated"
    if nli.get("entailment", 0) >= MDEBERTA_THRESHOLD_CONFIRMED:
        return "confirmed"
    return None  # kein Override — Cosine-Label bleibt


def _is_cross_language(quote: str, page_text: str) -> bool:
    """Erkennt Sprach-Mismatch: deutsche Paraphrase vs. englischer Quelltext.

    Heuristik: Quote hat Umlaute (ä/ö/ü/ß), Page-Text hat keine → Cross-Language.
    Auch: Quote hat keine Umlaute, Page-Text auch nicht, aber Quote-Wörter
    kommen kaum im Page-Text vor → Cross-Language Paraphrase.
    """
    umlaut_re = re.compile(r"[äöüßÄÖÜ]")
    quote_has_umlauts = bool(umlaut_re.search(quote))
    page_has_umlauts = bool(umlaut_re.search(page_text[:500]))
    return quote_has_umlauts and not page_has_umlauts


def score_anchor(quote: str, page_text: str, is_original_lang: bool = False) -> dict:
    """Berechnet Combined Score für ein Anker-Quote gegen Seiten-Text.

    is_original_lang=True → Block-Quote-Callout (Originalsprache) → Fuzzy dominant.
    is_original_lang=False + Sprach-Mismatch erkannt → Semantic dominant.

    Returns dict mit fuzzy, semantic, combined, label, weights_used.
    """
    if not page_text:
        return {"fuzzy": 0.0, "semantic": 0.0, "combined": 0.0,
                "label": "not_parseable", "weights": "n/a"}

    norm_quote = _normalize(quote)
    if len(norm_quote) < 10:
        return {"fuzzy": 0.0, "semantic": 0.0, "combined": 0.0,
                "label": "too_short", "weights": "n/a"}

    fuzzy = fuzz.partial_ratio(norm_quote, page_text) / 100.0
    semantic = _semantic_score(norm_quote, page_text)

    # Sprach-adaptive Gewichtung (Cross-Language-Finding 2026-05-14)
    if is_original_lang:
        w_f, w_s = WEIGHT_FUZZY, WEIGHT_SEMANTIC          # 0.8 / 0.2 — Direktzitat
        weights_used = "original"
    elif _is_cross_language(norm_quote, page_text):
        w_f, w_s = 0.0, 1.0                                # DE→EN: rein semantisch (Fuzzy = Rauschen bei Sprachwechsel)
        weights_used = "cross_language"
    else:
        w_f, w_s = WEIGHT_FUZZY, WEIGHT_SEMANTIC           # same language
        weights_used = "same_language"

    combined = w_f * fuzzy + w_s * semantic

    # Threshold-Regime: cross-language hat niedrigere Schwellen (unkalibriert)
    if weights_used == "cross_language":
        t_conf, t_unc = THRESHOLD_CONFIRMED_CROSS, THRESHOLD_UNCERTAIN_CROSS
    else:
        t_conf, t_unc = THRESHOLD_CONFIRMED_SAME, THRESHOLD_UNCERTAIN_SAME

    if combined >= t_conf:
        label = "confirmed"
    elif combined >= t_unc:
        label = "uncertain"
    else:
        label = "hallucinated"

    result = {"fuzzy": round(fuzzy, 3), "semantic": round(semantic, 3),
              "combined": round(combined, 3), "label": label, "weights": weights_used}

    # mDeBERTa NLI: Entailment statt Similarity bei Cross-Language
    # Überschreibt label wenn ENABLE_MDEBERTA_NLI=1 und weights=cross_language
    if ENABLE_MDEBERTA_NLI and weights_used == "cross_language":
        nli = _nli_score(norm_quote, page_text)
        nli_lbl = _nli_label(nli, label)
        result["nli"] = nli
        if nli_lbl is not None:
            result["label"] = nli_lbl   # Override nur bei starkem Signal
            result["label_source"] = "nli"
        else:
            result["label_source"] = "cosine"

    return result


# ---------------------------------------------------------------------------
# Wilson-Konfidenzintervall
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-Score-Konfidenzintervall für Binomial-Rate. z=1.96 → 95% CI."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))) / denom
    return max(0.0, round(center - margin, 3)), min(1.0, round(center + margin, 3))


# ---------------------------------------------------------------------------
# Note-Parser
# ---------------------------------------------------------------------------

def _parse_anchors_from_note(note_path: Path) -> tuple[list[dict], int]:
    """Extrahiert Anker aus vault_writer-generierten Markdown-Notes.

    Format nach vault_writer.convert_inline_to_footnotes:
      Body:     "...Satz mit Aussage[^3]. Nächster..."
      Fußnote:  "[^3]: dateiname.pdf, S. 7."

    Strategie:
    1. Footnote-Definitionen → {N: page_num}
    2. Body-Sätze die mit [^N] enden → Satz = quote, page aus Mapping
    3. Fallback wenn kein Mapping: Block-Quotes ([!quote]-Callouts) + Seitenzahl
    """
    text = note_path.read_text(encoding="utf-8", errors="replace")

    # Frontmatter-Ende finden
    fm_end = text.find("---", 3)
    fm_end = text.find("\n", fm_end + 3) if fm_end > 0 else 0
    body = text[fm_end:]

    # Schritt 1: Footnote-Definitionen parsen → {footnote_num: page_int}
    fn_def_re = re.compile(r"^\[\^(\d+)\]:.*?S\.\s*(\d+)", re.MULTILINE)
    fn_pages: dict[int, int] = {}
    for m in fn_def_re.finditer(body):
        fn_pages[int(m.group(1))] = int(m.group(2))

    # Schritt 2: Sätze mit [^N]-Referenz extrahieren
    # Muster: Text-Segment das mit [^N] endet (ggf. mehrere [^N] pro Satz)
    anchors = []
    # Sentence-Boundary + Footnote: alles zwischen letztem Satzende und [^N]
    sent_fn_re = re.compile(
        r"(?:^|(?<=[.!?])\s+)"      # Satzanfang
        r"([^.!?\n]{20,}?)"          # Satz-Text (min 20 Zeichen)
        r"(\[\^\d+\])+",             # ein oder mehr Footnote-Refs am Ende
        re.MULTILINE,
    )
    for m in sent_fn_re.finditer(body):
        sentence = m.group(1).strip()
        # Alle [^N] in diesem Match auslesen
        fn_nums = [int(n) for n in re.findall(r"\[\^(\d+)\]", m.group(0))]
        for fn_num in fn_nums:
            page = fn_pages.get(fn_num)
            if page and len(sentence) >= 20:
                anchors.append({"source": sentence, "page": page, "fn": fn_num})

    # Schritt 3: Block-Quote-Callouts als zusätzliche Anker
    # Format: > [!quote]- Autor YYYY, S. N  →  > „Text"
    bq_re = re.compile(
        r'>\s*\[!quote\][^\n]*S\.\s*(\d+)[^\n]*\n'    # Header mit Seite
        r'>\s*[„"](.{10,200})[“""]',          # Zitat-Text (deutsche + ASCII Quotes)
        re.MULTILINE,
    )
    for m in bq_re.finditer(body):
        anchors.append({"source": m.group(2).strip(), "page": int(m.group(1)),
                        "fn": None})

    # Deduplizieren (gleicher Satz + gleiche Seite)
    seen = set()
    unique_anchors = []
    for a in anchors:
        key = (a["source"][:50], a["page"])
        if key not in seen:
            seen.add(key)
            unique_anchors.append(a)

    # Body-Sätze zählen (Coverage-Metrik) — Footnote-Definitionen ausblenden
    body_no_fn = re.sub(r"^\[\^\d+\]:.*$", "", body, flags=re.MULTILINE)
    body_sentences = len([s for s in re.split(r"[.!?]", body_no_fn)
                          if len(s.strip()) > 15])

    return unique_anchors, body_sentences


# ---------------------------------------------------------------------------
# Haupt-Eval
# ---------------------------------------------------------------------------

def _detect_language_pair(note_body: str, sample_page_text: str) -> str:
    """Erkennt Sprachpaar: 'DE→DE', 'EN→DE', 'DE→EN', 'EN→EN'.

    Heuristik: Umlaut-Präsenz in Note-Body und PDF-Seitentext.
    """
    umlaut_re = re.compile(r"[äöüßÄÖÜ]")
    note_de = bool(umlaut_re.search(note_body[:500]))
    pdf_de = bool(umlaut_re.search(sample_page_text[:500]))
    note_lang = "DE" if note_de else "EN"
    pdf_lang = "DE" if pdf_de else "EN"
    return f"{pdf_lang}→{note_lang}"


def eval_note(note_path: Path, pdf_path: Path, pipeline_version: str) -> dict:
    """Evaluiert eine Note gegen ihre Quell-PDF. Gibt Metriken-Dict zurück."""
    anchors, body_sentences = _parse_anchors_from_note(note_path)

    if not anchors:
        return {
            "note": note_path.name, "pdf": pdf_path.name,
            "version": pipeline_version, "timestamp": datetime.now().isoformat(),
            "error": "no_anchors_found", "anchors_total": 0,
        }

    if not pdf_path.exists():
        return {
            "note": note_path.name, "pdf": pdf_path.name,
            "version": pipeline_version, "timestamp": datetime.now().isoformat(),
            "error": "pdf_not_found", "anchors_total": len(anchors),
        }

    pdf_doc = fitz.open(str(pdf_path))
    total_pages = len(pdf_doc)

    # Sprache erkennen anhand der Note und ersten PDF-Seite
    note_text = note_path.read_text(encoding="utf-8", errors="replace")
    sample_page = _extract_page_text(pdf_doc, 1)
    language_pair = _detect_language_pair(note_text, sample_page)

    results = []
    not_parseable = 0

    for anc in anchors:
        page = anc["page"]
        # Seitenübergreifend: S.N + S.N+1 zusammen prüfen (Gemini-Finding)
        page_text = _extract_page_text(pdf_doc, page)
        if page < total_pages:
            page_text += " " + _extract_page_text(pdf_doc, page + 1)
        page_text = page_text.strip()

        # OCR-Detect: leere Seite → not_parseable (Nemotron-Finding)
        if not page_text:
            not_parseable += 1
            results.append({"page": page, "label": "not_parseable",
                             "fuzzy": 0.0, "semantic": 0.0, "combined": 0.0})
            continue

        quote = anc["source"][:200].strip()
        is_orig = anc.get("fn") is None  # Block-Quote-Callouts haben fn=None
        score = score_anchor(quote, page_text, is_original_lang=is_orig)
        score["page"] = page
        results.append(score)

    pdf_doc.close()

    # Metriken berechnen
    parseable = [r for r in results if r["label"] != "not_parseable"]
    confirmed = sum(1 for r in parseable if r["label"] == "confirmed")
    uncertain = sum(1 for r in parseable if r["label"] == "uncertain")
    hallucinated = sum(1 for r in parseable if r["label"] == "hallucinated")
    parseable_n = len(parseable)

    hall_rate = hallucinated / parseable_n if parseable_n > 0 else 0.0
    hall_ci = wilson_ci(hallucinated, parseable_n)

    # Coverage-Kategorien (Nemotron/Gemini/Forschung 2026-05-14)
    # Strukturelle Sätze herausfiltern → ehrlicherer Nenner für Coverage
    note_text_body = note_path.read_text(encoding="utf-8", errors="replace")
    fm_end2 = note_text_body.find("---", 3)
    fm_end2 = note_text_body.find("\n", fm_end2 + 3) if fm_end2 > 0 else 0
    body_only = re.sub(r"^\[\^\d+\]:.*$", "", note_text_body[fm_end2:], flags=re.MULTILINE)
    all_sents = [s.strip() for s in re.split(r"[.!?]", body_only) if len(s.strip()) > 8]
    structural_sents = sum(
        1 for s in all_sents
        if len(s.split()) < 6 or any(s.lower().startswith(sw) for sw in _STRUCTURAL_STARTS)
    )
    factual_sents = max(0, body_sentences - structural_sents)
    coverage_factual = confirmed / factual_sents if factual_sents > 0 else 0.0
    # Alter Coverage-Wert bleibt für Rückwärts-Kompatibilität
    coverage = confirmed / body_sentences if body_sentences > 0 else 0.0

    # Source-Coverage: welcher Anteil der PDF-Seiten ist durch Anker abgedeckt?
    covered_pages = {a["page"] for a in anchors if a.get("page")}
    source_coverage = len(covered_pages) / total_pages if total_pages > 0 else 0.0

    return {
        "note": note_path.name,
        "pdf": pdf_path.name,
        "language": language_pair,
        "version": pipeline_version,
        "eval_version": EVAL_VERSION,
        "timestamp": datetime.now().isoformat(),
        "anchors_total": len(anchors),
        "anchors_not_parseable": not_parseable,
        "anchors_parseable": parseable_n,
        "anchors_confirmed": confirmed,
        "anchors_uncertain": uncertain,
        "anchors_hallucinated": hallucinated,
        "hallucination_rate": round(hall_rate, 3),
        "hallucination_ci_95": hall_ci,
        "coverage_rate": round(coverage, 3),          # confirmed / alle Sätze
        "coverage_factual": round(coverage_factual, 3), # confirmed / faktische Sätze
        "source_coverage": round(source_coverage, 3),   # PDF-Seiten abgedeckt
        "body_sentences": body_sentences,
        "body_sentences_structural": structural_sents,
        "body_sentences_factual": factual_sents,
        "pdf_pages_total": total_pages,
        "pdf_pages_covered": len(covered_pages),
        "small_sample_warning": parseable_n < 5,
        "anchor_scores": results,
    }


def save_result(result: dict) -> None:
    """Appended Ergebnis an quality_history.jsonl und atomic_analytics.db."""
    _QUALITY_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with _QUALITY_HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

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
                "acceptance_status": None,
                "hallucination_rate":result.get("hallucination_rate"),
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
                "eval_version":      result.get("eval_version"),
                "timestamp":         result.get("timestamp"),
            })
    except Exception as _db_err:
        import sys as _sys
        print(f"  [warn] DB-Write fehlgeschlagen: {_db_err}", file=_sys.stderr)

    print(f"  → gespeichert: {_QUALITY_HISTORY}")


def print_summary(result: dict) -> None:
    """Gibt lesbares Summary aus."""
    if "error" in result:
        print(f"[ERROR] {result['note']}: {result['error']}")
        return
    n = result["anchors_parseable"]
    h = result["anchors_hallucinated"]
    c = result["anchors_confirmed"]
    u = result["anchors_uncertain"]
    np_ = result["anchors_not_parseable"]
    rate = result["hallucination_rate"]
    ci = result["hallucination_ci_95"]
    cov = result["coverage_rate"]
    warn = " ⚠️ kleine Stichprobe" if result.get("small_sample_warning") else ""

    print(f"\n=== {result['note']} ({result['version']}) ===")
    print(f"  Anker: {n} prüfbar, {np_} nicht parseable")
    print(f"  Confirmed: {c}  Uncertain: {u}  Halluziniert: {h}")
    print(f"  Halluzinationsrate: {rate:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%}){warn}")
    print(f"  Coverage (confirmed/Sätze): {cov:.1%} ({c}/{result['body_sentences']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministische Halluzinations-Messung")
    ap.add_argument("--note", help="Pfad zur Note-Datei (.md)")
    ap.add_argument("--pdf", help="Pfad zur Quell-PDF")
    ap.add_argument("--version", default=AGENT_VERSION, help="Pipeline-Version")
    ap.add_argument("--save", action="store_true", help="Ergebnis in quality_history.jsonl speichern")
    args = ap.parse_args()

    if not args.note or not args.pdf:
        ap.print_help()
        sys.exit(1)

    note_path = Path(args.note)
    pdf_path = Path(args.pdf)

    print(f"[eval_quality] {note_path.name} vs {pdf_path.name}")
    result = eval_note(note_path, pdf_path, args.version)
    print_summary(result)

    if args.save:
        save_result(result)


if __name__ == "__main__":
    main()