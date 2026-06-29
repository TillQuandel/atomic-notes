"""Tests: justification aus dem Judge-Output entfernen (Output-Token-Reduktion).

Das `justification`-Feld fliesst in keine Decision/Metrik/AC1 (reporting-only) und
steht im Schema NACH dem Label (post-hoc, kein Chain-of-Thought). Es zu entfernen
spart ~44% der Judge-Output-Tokens.
Die AC1-Neutralitaet ist per separatem Label-Stabilitaets-A/B empirisch zu belegen
(NICHT Teil dieser Unit-Tests): das Entfernen der Few-Shot-Rationales aus dem
Prompt-Kontext koennte die Label-Kalibrierung theoretisch beeinflussen.
"""

from __future__ import annotations

import generative.eval_quality_v4 as eq


def test_judge_prompt_omits_justification():
    """Der Judge-Prompt (inkl. Few-Shots) fordert kein justification mehr an,
    damit das Modell das Feld nicht generiert."""
    for variant in ("primary", "audit"):
        prompt = eq._prompt_header("Ein Titel", variant=variant)
        assert "justification" not in prompt.lower()


def test_normalize_judge_rows_without_justification():
    """Parser akzeptiert Judge-Output ohne justification-Feld, haelt Label/Evidence
    intakt und fuehrt justification nicht mehr im normalisierten Row."""
    item = eq.RetrievedContext(
        claim_idx=0,
        claim="Eine Behauptung.",
        contexts=[{"chunk_idx": 0, "pages": [1], "text": "Kontext."}],
        top_cosine=0.5,
        best_chunk_idx=0,
        best_page=1,
    )
    raw = [
        {
            "claim_idx": 0,
            "label": "supported_paraphrase",
            "evidence": "ein woertliches Zitat aus dem Kontext",
            "best_page": 1,
        }
    ]

    rows, _flags = eq._normalize_judge_rows(raw, [item])

    assert len(rows) == 1
    row = rows[0]
    assert row["claim_idx"] == 0
    assert row["label"] == "supported_paraphrase"
    assert row["evidence"] == "ein woertliches Zitat aus dem Kontext"
    assert "justification" not in row


def test_normalize_judge_rows_ignores_legacy_justification():
    """Abwaertskompatibilitaet: alter/gecachter Judge-Output mit justification wird
    toleriert (Feld ignoriert), Label bleibt intakt, kein Parser-Bruch."""
    item = eq.RetrievedContext(
        claim_idx=0,
        claim="Eine Behauptung.",
        contexts=[{"chunk_idx": 0, "pages": [1], "text": "Kontext."}],
        top_cosine=0.5,
        best_chunk_idx=0,
        best_page=1,
    )
    raw = [
        {
            "claim_idx": 0,
            "label": "contradicted",
            "evidence": "ein Zitat",
            "justification": "veraltetes Feld aus altem Cache",
            "best_page": 1,
        }
    ]

    rows, _flags = eq._normalize_judge_rows(raw, [item])

    assert rows[0]["label"] == "contradicted"
    assert "justification" not in rows[0]
