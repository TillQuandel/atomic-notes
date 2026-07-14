# -*- coding: utf-8 -*-
"""Teilfix zu #151 (Nebenbefund PR #181): eval_note rief _claim_scores_from_judge
zweimal mit identischen Primaer-Inputs auf (Alt-Code: einmal ohne audit_rows, dann
nochmal MIT audit_rows aber wieder ueber ALLE judge_rows) -- fuer nicht-auditierte
Claims war der zweite Durchlauf bit-identische Doppelarbeit (reine CPU, kein
LLM-Call, da audit_by_idx.get(claim_idx) fuer sie in beiden Durchlaeufen None liefert).

Der Fix beschraenkt den zweiten Durchlauf auf die tatsaechlich auditierten Claims
(echte Teilmenge) und merged das Ergebnis in die claim_scores des ersten Durchlaufs
zurueck. Fuer nicht-auditierte Claims bleibt der Eintrag aus dem ersten Durchlauf
unveraendert -- bit-identisch zum Alt-Verhalten, da deren audit_label ohnehin immer
None war.

Zwei Testklassen:
- TestAuditPassRecomputesOnlyAuditedClaims: Spy-Zaehler auf _claim_scores_from_judge.
  RED (Alt-Code): zweiter Durchlauf deckt exakt dieselben Claim-Indizes ab wie der
  erste. GREEN (Fix): zweiter Durchlauf ist eine ECHTE Teilmenge.
- TestEvalNoteResultUnchanged: Aequivalenz-Regression -- der volle eval_note-Output
  (claim_scores, Aggregat-Kennzahlen, llm_usage) ist vor/nach dem Fix bit-identisch
  (bis auf `timestamp`, das per Design bei jedem Lauf aktuell ist). Die erwarteten
  Werte sind der eingefrorene Snapshot des UNVERAENDERTEN Codes auf origin/master
  gegen genau dieses synthetische Fixture (Claims mit UND ohne audit_rows-Eintrag).
"""

from __future__ import annotations

import fitz

from generative import eval_quality_v4 as eq

LOW_COSINE_IDX = 0
N_CLAIMS = 6

CLAIMS = [
    "Erstes Testclaim mit ausreichend Textlaenge fuer die Extraktion Alpha.",
    "Zweites Testclaim mit ausreichend Textlaenge fuer die Extraktion Beta.",
    "Drittes Testclaim mit ausreichend Textlaenge fuer die Extraktion Gamma.",
    "Viertes Testclaim mit ausreichend Textlaenge fuer die Extraktion Delta.",
    "Fuenftes Testclaim mit ausreichend Textlaenge fuer die Extraktion Epsilon.",
    "Sechstes Testclaim mit ausreichend Textlaenge fuer die Extraktion Zeta.",
]
assert len(CLAIMS) == N_CLAIMS


def _make_pdf(tmp_path, name: str = "quelle.pdf"):
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


def _make_note(tmp_path, name: str = "note.md"):
    note_path = tmp_path / name
    body = "# Testnote\n\n" + " ".join(CLAIMS) + "\n"
    note_path.write_text(body, encoding="utf-8")
    return note_path


def _fixed_retrieved(claims):
    """Deterministische Retrieval-Ergebnisse: Claim 0 hat absichtlich niedriges
    Cosine (< RETRIEVAL_LOW_COSINE=0.4) -> garantierter Audit-Trigger; die uebrigen
    liegen sicher darueber."""
    retrieved = []
    for i, claim in enumerate(claims):
        cosine = 0.1 if i == LOW_COSINE_IDX else 0.9
        retrieved.append(
            eq.RetrievedContext(
                claim_idx=i,
                claim=claim,
                contexts=[{"rank": 1, "chunk_idx": 0, "pages": [1], "cosine": cosine, "text": "Kontext"}],
                top_cosine=cosine,
                best_chunk_idx=0,
                best_page=1,
            )
        )
    return retrieved


def _fake_call_judge(note_title, items, *, variant, use_cache):
    """Ersetzt den echten LLM-Call. Primaer-Pass: alle Claims 'supported_paraphrase'.
    Audit-Pass: fuer den nicht-low-cosine-Claim bewusst 'contradicted' (staerkeres
    Label) -> beweist per rule_audit_stricter_override eine ECHTE Label-Aenderung
    durch Audit, damit der Aequivalenztest mehr prueft als ein reines No-op."""
    rows = []
    for item in items:
        if variant == "audit":
            label = eq.SUPPORTED_PARAPHRASE if item.claim_idx == LOW_COSINE_IDX else eq.CONTRADICTED
        else:
            label = eq.SUPPORTED_PARAPHRASE
        evidence = None if item.claim_idx == LOW_COSINE_IDX else "Wissensluecke"
        rows.append(
            {
                "claim_idx": item.claim_idx,
                "label": label,
                "original_judge_label": label,
                "evidence": evidence,
                "best_page": 1,
            }
        )
    meta = {"calls": 1, "input_tokens": 7, "output_tokens": 7, "cached_calls": 0, "quality_flags": []}
    return rows, meta


def _run_eval_note(tmp_path, monkeypatch, spy: list | None = None):
    eq._reset_pdf_caches()
    pdf_path = _make_pdf(tmp_path)
    note_path = _make_note(tmp_path)

    monkeypatch.setattr(eq, "_chunk_embeddings", lambda pdf_path, chunks: None)
    monkeypatch.setattr(
        eq, "_retrieve_claim_contexts", lambda claims, chunks, chunk_embs=None: _fixed_retrieved(claims)
    )
    monkeypatch.setattr(eq, "_call_judge", _fake_call_judge)

    if spy is not None:
        real_fn = eq._claim_scores_from_judge

        def wrapped(claims, retrieved, judge_rows, pdf_text, audit_rows=None, corpus_normalized=None):
            spy.append(
                {
                    "judge_idxs": sorted(row["claim_idx"] for row in judge_rows),
                    "has_audit": audit_rows is not None,
                }
            )
            return real_fn(claims, retrieved, judge_rows, pdf_text, audit_rows, corpus_normalized=corpus_normalized)

        monkeypatch.setattr(eq, "_claim_scores_from_judge", wrapped)

    # pipeline_version wird explizit fixiert (statt den Default AGENT_VERSION aus
    # config.py durchzureichen) -- so bleibt EXPECTED_RESULT["version"] unten stabil
    # gegenueber _auto_version_bump (bumpt AGENT_VERSION im Maintainer-Modus bei
    # jeder Pipeline-Code-Aenderung), ohne den config-Wert im Test zu duplizieren.
    # Konvention wie test_per_agent_tracking.py::test_aggregate_includes_model_config.
    return eq.eval_note(note_path, pdf_path, pipeline_version="v0.0.0", content_hash=None)


class TestAuditPassRecomputesOnlyAuditedClaims:
    def test_second_pass_covers_a_strict_subset_not_all_claims(self, tmp_path, monkeypatch):
        calls: list[dict] = []
        result = _run_eval_note(tmp_path, monkeypatch, spy=calls)

        assert result["claims_total"] == N_CLAIMS
        assert len(calls) == 2  # ein Primaer-Durchlauf + ein Audit-Durchlauf

        primary_call, audit_call = calls
        assert primary_call["has_audit"] is False
        assert primary_call["judge_idxs"] == list(range(N_CLAIMS))
        assert audit_call["has_audit"] is True

        # Kern des Fixes: der Audit-Durchlauf rechnet NICHT erneut ueber alle Claims
        # (bit-identische Doppelarbeit fuer nicht-auditierte Claims), sondern nur
        # ueber die tatsaechlich auditierten -- eine ECHTE Teilmenge.
        audit_idxs = set(audit_call["judge_idxs"])
        assert audit_idxs, "Test-Fixture muss mindestens einen Audit-Trigger ausloesen"
        assert audit_idxs < set(primary_call["judge_idxs"])


# ---------------------------------------------------------------------------
# Aequivalenz-Regression: eval_note-Output vor/nach dem Fix bit-identisch
# ---------------------------------------------------------------------------
#
# EXPECTED_RESULT ist der eingefrorene Snapshot des UNVERAENDERTEN Codes auf
# origin/master (Commit a8da0ec, vor diesem Fix) gegen genau dieses Fixture --
# erzeugt durch einen einmaligen Lauf von `_run_eval_note`, `timestamp` entfernt
# (per Design bei jedem Lauf `datetime.now()`, kein Teil der zu pruefenden
# Ergebnis-Semantik). Claim 0 (top_cosine 0.1 < RETRIEVAL_LOW_COSINE) und Claim 2
# (stratifiziertes Sample, hash-basiert -- deterministisch fuer dieses Fixture)
# sind die auditierten Claims; Claim 2 zeigt eine ECHTE Label-Aenderung durch
# Audit-Override (supported_paraphrase -> contradicted). Alle anderen Claims
# durchlaufen den Audit-Pfad nicht und muessen unveraendert aus dem Primaer-
# Durchlauf stammen.
EXPECTED_RESULT = {
    "note": "note.md",
    "pdf": "quelle.pdf",
    "language": "DE→DE",
    "version": "v0.0.0",  # explizit fixiert in _run_eval_note, s. Kommentar dort
    "eval_version": "4.2",  # #232: Bump 4.1->4.2 (Retrieval-Methodik F1+F2), Snapshot nachgezogen
    "content_hash": None,
    "claims_total": 6,
    "claims_supported_exact": 0,
    "claims_supported_paraphrase": 4,
    "claims_partially_supported": 0,
    "claims_not_in_context": 0,
    "claims_contradicted": 1,
    "claims_retrieval_or_parse_uncertain": 1,
    "claims_parse_error": 0,
    "parse_error_count": 0,
    "valid_claims": 5,
    "rate_valid": True,
    "confirmed_rate": 0.8,
    "partial_rate": 0.0,
    "retrieval_failure_rate": 0.167,
    "uncertain_rate": 0.167,
    "parse_error_rate": 0.0,
    "claim_support_rate": 0.8,
    "citation_verification_rate": 1.0,
    "claims_with_evidence": 5,
    "evidence_verified_count": 5,
    "anchors_total": 6,
    "anchors_confirmed": 4,
    "anchors_hallucinated": 1,
    "hallucination_rate": 0.2,
    "coverage_rate": 0.8,
    "hallucination_ci_95": (0.036, 0.624),
    "pdf_chunks_total": 1,
    "claim_scores": [
        {
            "claim_idx": 0,
            "claim": "Testnote Erstes Testclaim mit ausreichend Textlaenge fuer die Extraktion Alpha.",
            "label": "retrieval_or_parse_uncertain",
            "decision_source": "system",
            "original_judge_label": "supported_paraphrase",
            "label_original": "supported_paraphrase",
            "audit_label": "supported_paraphrase",
            "evidence": None,
            "evidence_verified": None,
            "evidence_verification_score": None,
            "best_page": 1,
            "top_cosine": 0.1,
            "best_chunk_idx": 0,
            "retrieved_contexts": [{"rank": 1, "chunk_idx": 0, "pages": [1], "cosine": 0.1, "text": "Kontext"}],
            "quality_flags": ["retrieval_low_cosine"],
        },
        {
            "claim_idx": 1,
            "claim": "Zweites Testclaim mit ausreichend Textlaenge fuer die Extraktion Beta.",
            "label": "supported_paraphrase",
            "decision_source": "primary",
            "original_judge_label": "supported_paraphrase",
            "label_original": None,
            "audit_label": None,
            "evidence": "Wissensluecke",
            "evidence_verified": True,
            "evidence_verification_score": 1.0,
            "best_page": 1,
            "top_cosine": 0.9,
            "best_chunk_idx": 0,
            "retrieved_contexts": [{"rank": 1, "chunk_idx": 0, "pages": [1], "cosine": 0.9, "text": "Kontext"}],
            "quality_flags": [],
        },
        {
            "claim_idx": 2,
            "claim": "Drittes Testclaim mit ausreichend Textlaenge fuer die Extraktion Gamma.",
            "label": "contradicted",
            "decision_source": "audit_override",
            "original_judge_label": "supported_paraphrase",
            "label_original": "supported_paraphrase",
            "audit_label": "contradicted",
            "evidence": "Wissensluecke",
            "evidence_verified": True,
            "evidence_verification_score": 1.0,
            "best_page": 1,
            "top_cosine": 0.9,
            "best_chunk_idx": 0,
            "retrieved_contexts": [{"rank": 1, "chunk_idx": 0, "pages": [1], "cosine": 0.9, "text": "Kontext"}],
            "quality_flags": sorted(["judge_uneinig", "audit_overridden"]),
        },
        {
            "claim_idx": 3,
            "claim": "Viertes Testclaim mit ausreichend Textlaenge fuer die Extraktion Delta.",
            "label": "supported_paraphrase",
            "decision_source": "primary",
            "original_judge_label": "supported_paraphrase",
            "label_original": None,
            "audit_label": None,
            "evidence": "Wissensluecke",
            "evidence_verified": True,
            "evidence_verification_score": 1.0,
            "best_page": 1,
            "top_cosine": 0.9,
            "best_chunk_idx": 0,
            "retrieved_contexts": [{"rank": 1, "chunk_idx": 0, "pages": [1], "cosine": 0.9, "text": "Kontext"}],
            "quality_flags": [],
        },
        {
            "claim_idx": 4,
            "claim": "Fuenftes Testclaim mit ausreichend Textlaenge fuer die Extraktion Epsilon.",
            "label": "supported_paraphrase",
            "decision_source": "primary",
            "original_judge_label": "supported_paraphrase",
            "label_original": None,
            "audit_label": None,
            "evidence": "Wissensluecke",
            "evidence_verified": True,
            "evidence_verification_score": 1.0,
            "best_page": 1,
            "top_cosine": 0.9,
            "best_chunk_idx": 0,
            "retrieved_contexts": [{"rank": 1, "chunk_idx": 0, "pages": [1], "cosine": 0.9, "text": "Kontext"}],
            "quality_flags": [],
        },
        {
            "claim_idx": 5,
            "claim": "Sechstes Testclaim mit ausreichend Textlaenge fuer die Extraktion Zeta.",
            "label": "supported_paraphrase",
            "decision_source": "primary",
            "original_judge_label": "supported_paraphrase",
            "label_original": None,
            "audit_label": None,
            "evidence": "Wissensluecke",
            "evidence_verified": True,
            "evidence_verification_score": 1.0,
            "best_page": 1,
            "top_cosine": 0.9,
            "best_chunk_idx": 0,
            "retrieved_contexts": [{"rank": 1, "chunk_idx": 0, "pages": [1], "cosine": 0.9, "text": "Kontext"}],
            "quality_flags": [],
        },
    ],
    "quality_flags": ["audit_overridden", "judge_uneinig", "retrieval_low_cosine"],
    "llm_usage": {"calls": 2, "input_tokens": 14, "output_tokens": 14, "cached_calls": 0},
    # model_config wird nicht als eigenes Literal dupliziert (eval_note hat dafuer
    # keinen Override-Parameter wie pipeline_version) -- stattdessen direkter Bezug
    # auf den echten Modulwert, damit Aenderungen an config.MODEL_CONFIG diesen
    # Snapshot nicht grundlos brechen. _aggregate() schreibt exakt dieses Objekt
    # unveraendert ins Ergebnis, die Bit-Identitaets-Aussage bleibt also erhalten.
    "model_config": eq.MODEL_CONFIG,
}


class TestEvalNoteResultUnchanged:
    def test_full_result_matches_pre_fix_snapshot(self, tmp_path, monkeypatch):
        result = _run_eval_note(tmp_path, monkeypatch, spy=None)
        result.pop("timestamp", None)
        # Pro-Claim quality_flags stammen aus einem frozenset (decision.flags in
        # _claim_scores_from_judge, unveraendert von diesem Fix) -- die Iterations-
        # reihenfolge haengt vom PYTHONHASHSEED ab (pro Prozess randomisiert), der
        # INHALT ist aber stabil. Fuer den bit-identisch-Vergleich wird deshalb nur
        # die Reihenfolge dieses einen Felds normalisiert, nicht die Werte selbst.
        for score in result["claim_scores"]:
            score["quality_flags"] = sorted(score["quality_flags"])

        assert result == EXPECTED_RESULT
