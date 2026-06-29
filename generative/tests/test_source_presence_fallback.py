# -*- coding: utf-8 -*-
"""Fix B (flag-only): markiert not_in_context-Claims als mögliche Retrieval-Misses,
OHNE das Label zu ändern — kein Masking.

Cross-Model-Review (Qwen+Mistral, 2026-06-28) + empirische Verifikation: reines
Cosine-Reklassifizieren würde Overgeneralization-Halluzinationen maskieren
(„Europa" → „alle Regionen", cos 0.845). Deshalb NUR flaggen: hallucination_rate
bleibt unverändert, die Kandidaten werden für Review/Kalibrierung sichtbar.
contradicted bleibt unberührt.
"""
from generative.eval_quality_v4 import apply_source_presence_fallback


def _score(label):
    return [{"claim": "irgendein Claim", "label": label, "quality_flags": []}]


def test_high_presence_flags_but_keeps_label():
    out = apply_source_presence_fallback(_score("not_in_context"), lambda c: 0.86, threshold=0.75)
    assert out[0]["label"] == "not_in_context"  # KEIN Relabel — kein Masking
    assert "possible_retrieval_miss" in out[0]["quality_flags"]
    assert out[0]["source_presence_score"] == 0.86


def test_low_presence_no_flag_but_score_recorded():
    out = apply_source_presence_fallback(_score("not_in_context"), lambda c: 0.58, threshold=0.75)
    assert out[0]["label"] == "not_in_context"
    assert "possible_retrieval_miss" not in (out[0].get("quality_flags") or [])
    assert out[0]["source_presence_score"] == 0.58  # Score immer gesetzt (Diagnostik/Kalibrierung)


def test_threshold_boundary_is_inclusive():
    out = apply_source_presence_fallback(_score("not_in_context"), lambda c: 0.75, threshold=0.75)
    assert "possible_retrieval_miss" in out[0]["quality_flags"]


def test_contradicted_is_untouched():
    out = apply_source_presence_fallback(_score("contradicted"), lambda c: 0.95, threshold=0.75)
    assert out[0]["label"] == "contradicted"
    assert "possible_retrieval_miss" not in (out[0].get("quality_flags") or [])
    assert "source_presence_score" not in out[0]


def test_supported_claims_untouched():
    out = apply_source_presence_fallback(_score("supported_exact"), lambda c: 0.99, threshold=0.75)
    assert out[0]["label"] == "supported_exact"
    assert "source_presence_score" not in out[0]


def test_none_score_no_flag_no_score():
    out = apply_source_presence_fallback(_score("not_in_context"), lambda c: None, threshold=0.75)
    assert out[0]["label"] == "not_in_context"
    assert "source_presence_score" not in out[0]
