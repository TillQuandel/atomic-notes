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
from generative.eval_quality_v4 import _build_context_pool, _normalize_for_evidence, _retrieve_claim_contexts
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
@pytest.mark.xfail(
    reason="#232 Retrieval-Fix (PR-B, F1+F2) steht aus -- Chunk-Granularitaet lässt "
    "den stuetzenden Satz oft ausserhalb der adaptive_k-Top-Chunks liegen.",
    strict=True,
)
def test_recall_false_positive_anchors(goldset, context_pools_by_pdf):
    """#232 Abnahme-Kriterium: Evidence-in-Pool-Recall ueber alle
    false_positive_retrieval_miss-Anker muss >= 90% sein. Auf master ist dieser
    Wert dokumentiert niedrig -- siehe PR-A-Bericht fuer die exakte Zahl. Sobald
    PR-B den Recall hebt, wird dieser Test unerwartet gruen (XPASS) und
    `strict=True` laesst den Lauf dann FAILEN, bis der xfail-Marker entfernt wird."""
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
