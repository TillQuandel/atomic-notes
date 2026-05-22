from __future__ import annotations

from decision_engine import ClaimDecision, Label
from decision_engine.aggregation import aggregate, confirmed_rate, hallucination_rate, label_counts


def decision(label: Label) -> ClaimDecision:
    return ClaimDecision(label, frozenset(), "primary")


def test_hallucination_rate_excludes_system_labels():
    decisions = [
        decision(Label.NOT_IN_CONTEXT),
        decision(Label.RETRIEVAL_UNCERTAIN),
        decision(Label.SUPPORTED_EXACT),
    ]
    metric = hallucination_rate(decisions)
    assert metric.valid
    assert metric.value == 0.5


def test_all_system_labels_returns_invalid_metric():
    metric = hallucination_rate([decision(Label.PARSE_ERROR)])
    assert not metric.valid
    assert metric.value == -1.0


def test_confirmed_rate_counts_exact_and_paraphrase_only():
    decisions = [
        decision(Label.SUPPORTED_EXACT),
        decision(Label.SUPPORTED_PARAPHRASE),
        decision(Label.PARTIALLY_SUPPORTED),
        decision(Label.RETRIEVAL_UNCERTAIN),
    ]
    metric = confirmed_rate(decisions)
    assert metric.valid
    assert metric.value == 0.667


def test_label_counts_includes_zeroes_for_all_labels():
    counts = label_counts([decision(Label.CONTRADICTED)])
    assert counts[Label.CONTRADICTED] == 1
    assert counts[Label.SUPPORTED_EXACT] == 0


def test_aggregate_returns_json_ready_metrics_and_counts():
    decisions = [
        decision(Label.SUPPORTED_EXACT),
        decision(Label.NOT_IN_CONTEXT),
        decision(Label.RETRIEVAL_UNCERTAIN),
        decision(Label.PARSE_ERROR),
    ]
    result = aggregate(decisions)
    assert result["claims_total"] == 4
    assert result["claims_supported_exact"] == 1
    assert result["claims_not_in_context"] == 1
    assert result["claims_retrieval_or_parse_uncertain"] == 1
    assert result["claims_parse_error"] == 1
    assert result["valid_claims"] == 2
    # v4.1: metrics nested mit per-Metric validity, kein irreführendes globales rate_valid mehr
    assert result["metrics"]["hallucination_rate"]["valid"] is True
    assert result["metrics"]["hallucination_rate"]["value"] == 0.5
    assert result["metrics"]["confirmed_rate"]["value"] == 0.5
    # Backward-Compat flache Felder (für Dashboard/Orchestrator)
    assert result["hallucination_rate"] == 0.5
    assert result["confirmed_rate"] == 0.5


def test_aggregate_empty_decisions_all_metrics_invalid():
    """v4.1 Fix 2: bei total=0 ALLE Rates konsistent invalid (-1.0)."""
    result = aggregate([])
    assert result["claims_total"] == 0
    assert result["valid_claims"] == 0
    # Alle Rates invalid
    for name in ("hallucination_rate", "confirmed_rate", "partial_rate",
                 "uncertain_rate", "parse_error_rate", "claim_support_rate"):
        assert result["metrics"][name]["valid"] is False
        assert result["metrics"][name]["value"] == -1.0


def test_aggregate_only_system_labels_hallucination_invalid():
    """Nur Systemlabels → hallucination_rate invalid, parse_error_rate/uncertain_rate valid."""
    decisions = [decision(Label.PARSE_ERROR), decision(Label.RETRIEVAL_UNCERTAIN)]
    result = aggregate(decisions)
    # hallucination_rate: keine valid_claims → invalid
    assert result["metrics"]["hallucination_rate"]["valid"] is False
    # parse_error/uncertain_rate sind über total → valid
    assert result["metrics"]["parse_error_rate"]["valid"] is True
    assert result["metrics"]["parse_error_rate"]["value"] == 0.5
    assert result["metrics"]["uncertain_rate"]["value"] == 0.5
