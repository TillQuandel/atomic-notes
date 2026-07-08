# -*- coding: utf-8 -*-
"""Stage-8-PDF-Memoisierung (#151, Punkt 1+2).

eval_note oeffnete dieselbe Quell-PDF bis 3x pro Note (build_chunks,
_verification_text, _build_presence_scorer), re-encodete alle Chunk-Embeddings pro
Note und normalisierte den GESAMTEN Volltext per Regex pro Claim. Der Prozess-Cache
oeffnet/encodet/normalisiert genau einmal pro (PDF, Lauf).

Zwei Testklassen:
- Zaehler: fitz.open == 1 ueber mehrere Notes; Chunk-Encoding == 1; Corpus-
  Normalisierung == 1 statt pro Claim.
- Identitaet: die memoisierten Zwischenwerte sind byte-/wertgleich zur
  Nicht-memoisierten Berechnung (Eval ist kalibrierungs-sensibel — kein Drift).
"""

from __future__ import annotations

import fitz
import numpy as np

from generative import eval_quality_v4 as eq
from generative import eval_quality_v2 as eq2


def _make_pdf(tmp_path, name="quelle.pdf"):
    pdf_path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    text = "\n".join(
        [
            "Information seeking is the purposive acquisition of knowledge from many sources.",
            "Wilson beschreibt Informationsverhalten als uebergeordnetes Rahmenkonzept.",
            "Das ISP-Modell von Kuhlthau umfasst sechs aufeinanderfolgende Phasen.",
            "Berrypicking betont die iterative Natur realer Suchprozesse deutlich.",
            "Relevanzurteile veraendern sich im Verlauf einer laengeren Recherche.",
            "Serendipity spielt bei der Entdeckung unerwarteter Quellen eine Rolle.",
            "Der Anomalous State of Knowledge beschreibt eine erkannte Wissensluecke.",
            "Nutzer formulieren ihren Informationsbedarf oft nur unvollstaendig aus.",
        ]
    )
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class _FakeModel:
    """Deterministisch (pro Text stabiler Vektor), zaehlt encode-Aufrufe."""

    def __init__(self, counter: dict):
        self._counter = counter

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        self._counter["encode"] = self._counter.get("encode", 0) + 1
        out = []
        for t in texts:
            h = abs(hash(t)) % 997
            v = np.array([h % 7 + 1.0, h % 5 + 1.0, h % 3 + 1.0, 1.0])
            out.append(v / np.linalg.norm(v))
        return np.array(out)


# ---------------------------------------------------------------------------
# Zaehler-Tests (RED vor Memo: N Oeffnungen/Normalisierungen; GREEN: 1)
# ---------------------------------------------------------------------------


class TestOpenAndEncodeCounters:
    def test_pdf_opened_once_across_three_notes(self, tmp_path, monkeypatch):
        # Guard: auf Alt-Code (RED-Messung) existiert _reset_pdf_caches nicht.
        getattr(eq, "_reset_pdf_caches", lambda: None)()
        pdf = _make_pdf(tmp_path)

        # Modell stubben, damit der Praesenz-Scorer kein echtes MiniLM laedt.
        monkeypatch.setattr(eq, "_model", lambda: _FakeModel({}))

        calls = {"open": 0}
        real_open = fitz.open

        def counting_open(*a, **k):
            calls["open"] += 1
            return real_open(*a, **k)

        monkeypatch.setattr(fitz, "open", counting_open)

        # Drei Notes derselben PDF: jede braucht Volltext + Praesenz-Saetze.
        not_in_context = [{"label": "not_in_context", "claim": "irgendein Claim"}]
        for _ in range(3):
            _ = eq._verification_text(pdf)
            scorer = eq._build_presence_scorer(pdf, not_in_context)
            scorer("irgendein Claim")

        assert calls["open"] == 1  # RED (Alt-Code): 6 (3x Volltext + 3x Praesenz)

    def test_chunk_embeddings_encoded_once_per_pdf(self, tmp_path, monkeypatch):
        eq._reset_pdf_caches()
        pdf = _make_pdf(tmp_path)
        chunks = eq._pdf_artifacts(pdf).chunks
        assert chunks  # PDF ist parsebar

        counter: dict = {}
        monkeypatch.setattr(eq, "_model", lambda: _FakeModel(counter))

        e1 = eq._chunk_embeddings(pdf, chunks)
        e2 = eq._chunk_embeddings(pdf, chunks)

        assert counter["encode"] == 1  # zweiter Aufruf: Cache-Hit
        assert e1 is e2

    def test_evidence_corpus_normalized_once_not_per_claim(self, tmp_path, monkeypatch):
        pdf_text = "Der Anomalous State of Knowledge beschreibt eine erkannte Wissensluecke im Detail."

        orig_norm = eq._normalize_for_evidence
        seen: list[str] = []

        def spy(text):
            seen.append(text)
            return orig_norm(text)

        monkeypatch.setattr(eq, "_normalize_for_evidence", spy)

        # Fuenf Claims, jeder mit Evidence — Alt-Code normalisierte den Corpus pro Claim.
        n = 5
        claims = [f"Claim Nummer {i} mit ausreichend Textlaenge." for i in range(n)]
        retrieved = [
            eq.RetrievedContext(
                claim_idx=i,
                claim=claims[i],
                contexts=[{"rank": 1, "chunk_idx": 0, "pages": [1], "cosine": 0.9, "text": "kontext"}],
                top_cosine=0.9,
                best_chunk_idx=0,
                best_page=1,
            )
            for i in range(n)
        ]
        judge_rows = [
            {
                "claim_idx": i,
                "label": eq.SUPPORTED_PARAPHRASE,
                "original_judge_label": eq.SUPPORTED_PARAPHRASE,
                "evidence": "Wissensluecke",
                "best_page": 1,
            }
            for i in range(n)
        ]

        eq._claim_scores_from_judge(claims, retrieved, judge_rows, pdf_text)

        corpus_calls = [t for t in seen if t == pdf_text]
        assert len(corpus_calls) == 1  # RED (Alt-Code): 5 (einmal pro Claim)


# ---------------------------------------------------------------------------
# Identitaets-Tests (memoisiert == nicht-memoisiert — kein Eval-Drift)
# ---------------------------------------------------------------------------


class TestMemoIdentity:
    def test_full_text_identical_to_fresh_open(self, tmp_path):
        eq._reset_pdf_caches()
        pdf = _make_pdf(tmp_path)

        with fitz.open(str(pdf)) as doc:
            fresh = " ".join(eq._extract_page_text(doc, p) for p in range(1, len(doc) + 1))

        assert eq._pdf_artifacts(pdf).full_text == fresh
        assert eq._verification_text(pdf) == fresh

    def test_chunks_identical_to_build_chunks(self, tmp_path):
        eq._reset_pdf_caches()
        pdf = _make_pdf(tmp_path)

        fresh_chunks = eq2.build_chunks(pdf)
        memo_chunks = eq._pdf_artifacts(pdf).chunks

        assert [(c.idx, c.text, c.pages) for c in memo_chunks] == [(c.idx, c.text, c.pages) for c in fresh_chunks]

    def test_sentences_identical_to_fresh_pdf_sentences(self, tmp_path):
        eq._reset_pdf_caches()
        pdf = _make_pdf(tmp_path)

        with fitz.open(str(pdf)) as doc:
            fresh = [s for s, _ in eq2._pdf_sentences(doc)]

        assert eq._pdf_artifacts(pdf).sentences == fresh

    def test_chunk_embeddings_identical_to_direct_encode(self, tmp_path, monkeypatch):
        eq._reset_pdf_caches()
        pdf = _make_pdf(tmp_path)
        chunks = eq._pdf_artifacts(pdf).chunks

        fake = _FakeModel({})
        monkeypatch.setattr(eq, "_model", lambda: fake)

        memo = eq._chunk_embeddings(pdf, chunks)
        direct = fake.encode([c.text for c in chunks])

        assert np.array_equal(memo, direct)

    def test_evidence_corpus_identical_and_cached(self, tmp_path):
        eq._reset_pdf_caches()
        pdf = _make_pdf(tmp_path)
        pdf_text = eq._verification_text(pdf)

        expected = eq._normalize_for_evidence(pdf_text)
        first = eq._evidence_corpus(pdf, pdf_text)
        second = eq._evidence_corpus(pdf, pdf_text)

        assert first == expected
        assert first is second  # Cache-Hit statt Neu-Normalisierung

    def test_verify_evidence_wrapper_matches_normalized_path(self):
        pdf_text = "Der Anomalous State of Knowledge beschreibt eine erkannte Wissensluecke."
        corpus = eq._normalize_for_evidence(pdf_text)

        assert eq._verify_evidence("Wissensluecke", pdf_text) == eq._verify_evidence_normalized("Wissensluecke", corpus)
        assert eq._verify_evidence(None, pdf_text) == (None, None)
