from __future__ import annotations



from decision_engine import ClaimDecision, ClaimInput, DEFAULT_CONFIG, Label, determine_decision
from decision_engine.aggregation import aggregate
from decision_engine.models import QualityFlag


def test_low_cosine_becomes_retrieval_uncertain() -> None:
    decision = determine_decision(
        ClaimInput(
            primary_label=Label.SUPPORTED_EXACT,
            audit_label=None,
            cosine=DEFAULT_CONFIG.retrieval_low_cosine_threshold - 0.001,
            evidence_verified=True,
            parse_failed=False,
        )
    )

    assert decision.label is Label.RETRIEVAL_UNCERTAIN
    assert decision.source == "system"
    assert QualityFlag.RETRIEVAL_LOW_COSINE in decision.flags


def test_unverified_exact_evidence_is_downgraded_to_not_in_context() -> None:
    decision = determine_decision(
        ClaimInput(
            primary_label=Label.SUPPORTED_EXACT,
            audit_label=None,
            cosine=0.9,
            evidence_verified=False,
            parse_failed=False,
        )
    )

    assert decision.label is Label.NOT_IN_CONTEXT
    assert decision.source == "downgrade"
    assert decision.flags == frozenset(
        {QualityFlag.EVIDENCE_UNVERIFIED, QualityFlag.EVIDENCE_FABRICATED}
    )


def test_audit_can_only_override_with_stricter_judge_label() -> None:
    decision = determine_decision(
        ClaimInput(
            primary_label=Label.SUPPORTED_PARAPHRASE,
            audit_label=Label.CONTRADICTED,
            cosine=0.9,
            evidence_verified=True,
            parse_failed=False,
        )
    )

    assert decision.label is Label.CONTRADICTED
    assert decision.source == "audit_override"
    assert QualityFlag.JUDGE_UNEINIG in decision.flags
    assert QualityFlag.AUDIT_OVERRIDDEN in decision.flags


def test_parse_failure_stays_parse_error_even_with_low_cosine() -> None:
    decision = determine_decision(
        ClaimInput(
            primary_label=Label.PARSE_ERROR,
            audit_label=Label.SUPPORTED_EXACT,
            cosine=0.0,
            evidence_verified=None,
            parse_failed=True,
        )
    )

    assert decision.label is Label.PARSE_ERROR
    assert decision.source == "system"
    assert QualityFlag.RETRIEVAL_LOW_COSINE not in decision.flags


def test_aggregate_exposes_eval_quality_v4_metrics() -> None:
    decisions = [
        ClaimDecision(Label.SUPPORTED_EXACT, frozenset(), "primary"),
        ClaimDecision(Label.SUPPORTED_PARAPHRASE, frozenset(), "primary"),
        ClaimDecision(Label.PARTIALLY_SUPPORTED, frozenset(), "primary"),
        ClaimDecision(Label.NOT_IN_CONTEXT, frozenset(), "primary"),
        ClaimDecision(Label.CONTRADICTED, frozenset(), "primary"),
        ClaimDecision(Label.RETRIEVAL_UNCERTAIN, frozenset(), "system"),
        ClaimDecision(Label.PARSE_ERROR, frozenset(), "system"),
    ]

    result = aggregate(decisions)

    assert result["claims_total"] == 7
    assert result["valid_claims"] == 5
    assert result["claims_retrieval_or_parse_uncertain"] == 1
    assert result["claims_parse_error"] == 1
    assert result["anchors_confirmed"] == 2
    assert result["anchors_hallucinated"] == 2
    assert result["hallucination_rate"] == 0.4
    assert result["confirmed_rate"] == 0.4
    assert result["claim_support_rate"] == 0.6
    assert result["metrics"]["hallucination_rate"] == {"value": 0.4, "valid": True}
