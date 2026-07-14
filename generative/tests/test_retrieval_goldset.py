"""Retrieval-Goldset-Harness fuer #232 (Granularitaets-Mismatch Chunk- vs. Satz-Retrieval).

Kein LLM-Call -- reine Embedding-/Chunking-/Retrieval-Rekonstruktion gegen die
echten Quell-PDFs, deterministisch (MiniLM + normalisierte Cosine sind stabil
ueber Wiederholungslaeufe). Nutzt exakt die Produktions-Bausteine aus
`eval_common`/`eval_quality_v4` (`_chunks_from_sentences`, `_pdf_sentences`,
`_retrieve_claim_contexts`, `_build_context_pool`, `_normalize_for_evidence`) --
keine Duplikat-Logik.

Goldset: `generative/calibration/retrieval-goldset/anchors.jsonl` (siehe README
dort fuer Herkunft/Adjudikations-Legende). 11 hand-adjudizierte Anker aus 3
Diagnose-Laeufen (Hrastinski 2008, Bates 2017, Suehl-Strohmenger 2008),
pdftotext-verifiziert.

Metrik: **Evidence-in-Pool-Recall** -- Anteil der `false_positive_retrieval_miss`-
Anker, deren `evidence_quote` (normalisiert) als Substring im Kontext-Pool
auftaucht, der der echten Pipeline fuer diesen Claim an den Judge geliefert
wuerde. Auf master ist dieser Recall NIEDRIG (dokumentierter Bug-Zustand aus
#232) -- die scharfe Assertion `recall >= 0.90` ist deshalb bewusst
`xfail(strict=True)`: sie bleibt gruen (erwarteter Fehlschlag) bis PR-B (#232
Retrieval-Fix, F1+F2) den Recall hebt: Nach dem Fix flippt der Test auf
unerwartet-gruen (XPASS) und `strict=True` laesst den Lauf dann FAILEN -- das
zwingt dazu, den xfail-Marker zu entfernen und das Abnahme-Kriterium scharf zu
stellen, statt es leise gruen zu belassen.

Negativ-Kontrolle (idx8, Hrastinski Table-3-Attributionsfehler): ein ECHTER
Fehler (kein Retrieval-Miss), MUSS ausserhalb der Recall-Population bleiben,
damit der spaetere Fix sie nicht wegkalibriert. Dieser Teil ist bereits heute
gruen (kein xfail).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import fitz
import pytest

from generative.config import LITERATURE_DIR
from generative.eval_common import _chunks_from_sentences, _pdf_sentences
from generative.eval_quality_v4 import (
    _build_context_pool,
    _normalize_for_evidence,
    _rescue_budget,
    _retrieve_claim_contexts,
)
from generative.pipeline.pdf_chunker import anchor_page_numbers

GOLDSET_PATH = Path(__file__).parent.parent / "calibration" / "retrieval-goldset" / "anchors.jsonl"
RECALL_THRESHOLD = 0.90


def _load_goldset() -> list[dict]:
    with GOLDSET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _evidence_in_pool(evidence_quote: str, pool_text: str) -> bool:
    """Substring-Check nach der EXAKT gleichen Normalisierung, die die Pipeline
    fuer die Zitat-Verifikation nutzt (`_verify_evidence` in eval_quality_v4)."""
    return _normalize_for_evidence(evidence_quote) in _normalize_for_evidence(pool_text)


@pytest.fixture(scope="module")
def goldset() -> list[dict]:
    records = _load_goldset()
    assert records, f"Goldset leer oder nicht gefunden: {GOLDSET_PATH}"
    return records


@pytest.fixture(scope="module")
def minilm_model():
    """Laedt MiniLM einmal fuers Modul. #232-Auftrag: bei RAM-Druck (WinError 1455 /
    OOM) NICHT abbrechen, sondern alle abhaengigen Tests mit Begruendung skippen --
    `load_ml_model` (generative/embeddings.py) uebersetzt den Low-Level-Fehler
    bereits in eine actionable RuntimeError-Message."""
    from generative.embeddings import _model

    try:
        return _model()
    except RuntimeError as exc:
        pytest.skip(f"MiniLM nicht ladbar (vermutlich RAM-Druck) -- Goldset-Recall-Test uebersprungen: {exc}")


@pytest.fixture(scope="module")
def context_pools_by_pdf(goldset, minilm_model) -> dict[str, str]:
    """Baut je Quell-PDF GENAU EINEN gemeinsamen Kontext-Pool ueber alle Goldset-
    Claims dieser PDF -- so wie die echte Pipeline alle Claims EINER Note gemeinsam
    an den Judge poolt (`_build_context_pool` in eval_quality_v4._call_judge). Alle
    Anker je PDF in diesem Goldset stammen aus derselben Note, das ist also keine
    Vereinfachung gegenueber Produktion."""
    by_pdf: dict[str, list[dict]] = defaultdict(list)
    for record in goldset:
        by_pdf[record["source_pdf"]].append(record)

    pools: dict[str, str] = {}
    for pdf_name, records in by_pdf.items():
        pdf_path = LITERATURE_DIR / pdf_name
        if not pdf_path.exists():
            pytest.skip(f"Quell-PDF fehlt (LITERATURE_DIR={LITERATURE_DIR}): {pdf_path}")
        with fitz.open(str(pdf_path)) as doc:
            page_numbers = anchor_page_numbers(pdf_path, len(doc))
            sentence_pairs = _pdf_sentences(doc, page_numbers)
        chunks = _chunks_from_sentences(sentence_pairs)
        claims = [record["claim"] for record in records]
        items = _retrieve_claim_contexts(claims, chunks)
        _, pool_text = _build_context_pool(items)
        pools[pdf_name] = pool_text
    return pools


def test_goldset_negative_control_metadata(goldset):
    """Metadaten-Invariante, modellunabhaengig: idx8 (Hrastinski Table-3-
    Attributionsfehler) ist ein ECHTER Fehler, kein Retrieval-Miss -- muss immer
    als `contradicted`/`true_hallucination` gefuehrt werden und darf nie in die
    false_positive_retrieval_miss-Recall-Population rutschen. Sonst wuerde ein
    kuenftiger Retrieval-Fix (PR-B) diesen Anker faelschlich mit-„reparieren"."""
    idx8 = next((r for r in goldset if r["id"] == "hrastinski-2008__sync__idx8"), None)
    assert idx8 is not None, "Negativ-Kontrolle idx8 fehlt im Goldset"
    assert idx8["expected_label"] == "contradicted"
    assert idx8["adjudication"] == "true_hallucination"

    fp_ids = {r["id"] for r in goldset if r["adjudication"] == "false_positive_retrieval_miss"}
    assert idx8["id"] not in fp_ids, "Negativ-Kontrolle darf nicht in die FP-Recall-Population fallen"


def _load_by_id(goldset: list[dict], anchor_id: str) -> dict:
    rec = next((r for r in goldset if r["id"] == anchor_id), None)
    assert rec is not None, f"Anker fehlt im Goldset: {anchor_id}"
    return rec


def _chunks_for(pdf_name: str):
    pdf_path = LITERATURE_DIR / pdf_name
    if not pdf_path.exists():
        pytest.skip(f"Quell-PDF fehlt (LITERATURE_DIR={LITERATURE_DIR}): {pdf_path}")
    with fitz.open(str(pdf_path)) as doc:
        page_numbers = anchor_page_numbers(pdf_path, len(doc))
        sentence_pairs = _pdf_sentences(doc, page_numbers)
    return _chunks_from_sentences(sentence_pairs)


def test_masking_probes_excluded_from_fp_population(goldset):
    """Metadaten-Invariante (modellunabhaengig): die Masking-Proben duerfen NIE in
    die false_positive_retrieval_miss-Recall-Population rutschen -- sonst wuerde der
    Recall-Test sie faelschlich als „einzubringende Belege" werten, obwohl sie gar
    keinen echten Beleg haben."""
    fp_ids = {r["id"] for r in goldset if r["adjudication"] == "false_positive_retrieval_miss"}
    for anchor_id, expected_adj in (
        ("hrastinski-2008__sync__masking-probe-dropout", "topical_hallucination_probe"),
        ("hrastinski-2008__sync__off-topic-control", "off_topic_control"),
    ):
        rec = _load_by_id(goldset, anchor_id)
        assert rec["adjudication"] == expected_adj
        assert rec["expected_label"] == "not_in_context"
        assert rec["id"] not in fp_ids


@pytest.mark.slow
def test_masking_probe_injects_context_but_not_fabricated_evidence(goldset, minilm_model):
    """#232 Masking-Richtung (Unter-Zaehl-Seite), testbar OHNE Judge:

    Eine THEMEN-NAHE, aber im PDF NICHT belegte Halluzination
    (`topical_hallucination_probe`) passiert den RESCUE_SENTENCE_MIN_COSINE-Floor
    (Off-Topic-Filter, kein Halluzinations-Filter) und bekommt on-topic Chunks in
    den Pool injiziert. Assertiert wird das, was ein Test OHNE LLM belegen KANN:

      1. Der (fabrizierte, nicht existierende) Beleg taucht NICHT im Pool auf --
         der Rescue erfindet keine Evidenz, er verschiebt nur Retrieval.
      2. Es WIRD aber on-topic Kontext injiziert (rescued > 0) -- genau das ist die
         Masking-Angriffsflaeche, die F1 (anders als das flag-only Fix B) oeffnet.
      3. Das Budget-Cap greift: rescued <= `_rescue_budget(n_chunks)`.

    Was dieser Test NICHT beweist: dass die aggregierte hallucination_rate ehrlich
    bleibt (d.h. dass der echte Judge den injizierten on-topic-Kontext NICHT lenient
    als Support liest). Diese volle Masking-Validierung braucht einen Re-Eval-Sweep
    mit echtem Judge (LLM, separat) -- die deterministischen Goldset-Tests fahren
    keinen Judge."""
    probe = _load_by_id(goldset, "hrastinski-2008__sync__masking-probe-dropout")
    chunks = _chunks_for(probe["source_pdf"])
    budget = _rescue_budget(len(chunks))

    items = _retrieve_claim_contexts([probe["claim"]], chunks)
    assert len(items) == 1
    item = items[0]
    _, pool_text = _build_context_pool(items)

    n_rescued = sum(1 for ctx in item.contexts if ctx.get("rescued"))
    # 1. Der fabrizierte Beleg darf NICHT im Pool erscheinen (kein erfundener Support).
    assert not _evidence_in_pool(probe["evidence_quote"], pool_text), (
        "Masking-Probe: der (nicht existierende) Beleg darf nicht im Pool auftauchen -- "
        "der Rescue darf keine Evidenz fabrizieren."
    )
    # 2. Es wird on-topic Kontext injiziert -- die dokumentierte Masking-Flaeche.
    assert n_rescued > 0, (
        "Masking-Probe: die themen-nahe Halluzination sollte den Off-Topic-Floor passieren "
        "und Kontext injiziert bekommen (sonst waere Assertion 1 vakuum)."
    )
    # 3. Das Budget-Cap begrenzt die Masking-Flaeche.
    assert n_rescued <= budget, f"Rescue-Budget verletzt: {n_rescued} > {budget}"


@pytest.mark.slow
def test_off_topic_claim_receives_no_rescue(goldset, minilm_model):
    """Zeigt, was der RESCUE_SENTENCE_MIN_COSINE-Floor TATSAECHLICH leistet (und nur
    das): reines OFF-TOPIC (Photosynthese gegen ein E-Learning-PDF, Top-Cosine ~0.08)
    bekommt NULL Rescue-Chunks. Der Floor ist ein Off-Topic-Gate, kein
    Halluzinations-Gate -- der Kontrast zur `topical_hallucination_probe` (die trotz
    fehlenden Belegs Kontext bekommt) macht genau diese Grenze testbar."""
    control = _load_by_id(goldset, "hrastinski-2008__sync__off-topic-control")
    chunks = _chunks_for(control["source_pdf"])
    items = _retrieve_claim_contexts([control["claim"]], chunks)
    assert len(items) == 1
    n_rescued = sum(1 for ctx in items[0].contexts if ctx.get("rescued"))
    assert n_rescued == 0, f"Off-Topic-Claim sollte 0 Rescue-Chunks bekommen, bekam {n_rescued}"


@pytest.mark.slow
def test_negative_control_idx8_evidence_reaches_pool(goldset, context_pools_by_pdf):
    """Anders als bei den FP-Ankern IST die kontrahierende Table-3-Evidenz fuer
    idx8 heute schon im Kontext-Pool (Retrieval hat hier funktioniert -- der Fehler
    liegt im Judge/Extraktions-Schritt, nicht im Retrieval). Deshalb kein xfail:
    dieser Teil ist bereits gruen und soll es bleiben, damit ein kuenftiger
    Retrieval-Fix nicht versehentlich genau diesen (bereits korrekten) Pfad
    veraendert."""
    idx8 = next(r for r in goldset if r["id"] == "hrastinski-2008__sync__idx8")
    pool_text = context_pools_by_pdf[idx8["source_pdf"]]
    assert _evidence_in_pool(idx8["evidence_quote"], pool_text), (
        "idx8: kontrahierende Table-3-Evidenz sollte im Kontext-Pool auffindbar sein "
        "(Retrieval funktioniert hier bereits -- Diagnose siehe README)."
    )


@pytest.mark.slow
def test_recall_false_positive_anchors(goldset, context_pools_by_pdf):
    """#232 Abnahme-Kriterium: Evidence-in-Pool-Recall ueber alle
    false_positive_retrieval_miss-Anker muss >= 90% sein. Vor PR-B (F1+F2) lag der
    Wert dokumentiert niedrig (2/10-4/10, umgebungsabhaengig -- siehe README). PR-B
    (Satz-Level-Retrieval-Rescue + Titel-Chunk-Deprioritisierung +
    Zitat-Marker-robuste Evidence-Normalisierung) hebt ihn ueber die Schwelle;
    der ehemalige `xfail(strict=True)`-Marker wurde entfernt und die Assertion ist
    jetzt hart. Die absolute Recall-Zahl driftet mit dem Embedding-/Library-Zustand
    (README-Caveat) -- die Abnahme prueft die Verbesserung in DERSELBEN Umgebung."""
    fp_records = [r for r in goldset if r["adjudication"] == "false_positive_retrieval_miss"]
    assert fp_records, "Kein false_positive_retrieval_miss-Anker im Goldset"

    hits = 0
    misses: list[str] = []
    for record in fp_records:
        pool_text = context_pools_by_pdf[record["source_pdf"]]
        if _evidence_in_pool(record["evidence_quote"], pool_text):
            hits += 1
        else:
            misses.append(record["id"])

    recall = hits / len(fp_records)
    print(
        f"\n[#232 Evidence-in-Pool-Recall] {hits}/{len(fp_records)} = {recall:.3f} "
        f"(Schwelle {RECALL_THRESHOLD}) -- Misses: {misses}"
    )
    assert recall >= RECALL_THRESHOLD, (
        f"Evidence-in-Pool-Recall {recall:.3f} ({hits}/{len(fp_records)}) unter Schwelle "
        f"{RECALL_THRESHOLD} -- Misses: {misses}"
    )
