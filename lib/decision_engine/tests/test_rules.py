from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

try:
    import hypothesis.strategies as st
    from hypothesis import given
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    given = None
    st = None

from decision_engine import ClaimInput, Label, determine_decision
from decision_engine.rules import DEFAULT_CONFIG
from decision_engine.models import ClaimDecision, QualityFlag, SYSTEM_LABELS


LABELS = list(Label)
OPTIONAL_LABELS = [None, *LABELS]
COSINES = [0.0, 0.399, 0.4, 0.401, 0.75, 1.0]
EVIDENCE_VALUES = [None, False, True]


def property_cases():
    for primary in LABELS:
        for audit in OPTIONAL_LABELS:
            for cosine in COSINES:
                for evidence in EVIDENCE_VALUES:
                    yield primary, audit, cosine, evidence


def hypothesis_or_cases(fn):
    if given is not None:
        return given(
            st.sampled_from(LABELS),
            st.one_of(st.none(), st.sampled_from(LABELS)),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.one_of(st.none(), st.booleans()),
        )(fn)
    return pytest.mark.parametrize("primary,audit,cosine,evidence", list(property_cases()))(fn)


@hypothesis_or_cases
def test_parse_error_terminal(primary, audit, cosine, evidence):
    # CLAUDE-PATTERN: parse_failed=True erfordert evidence_verified=None
    # (logische Konsistenz, von ClaimInput.__post_init__ erzwungen).
    if evidence is not None:
        return
    inp = ClaimInput(primary, audit, cosine, None, parse_failed=True)
    decision = determine_decision(inp)
    assert decision.label == Label.PARSE_ERROR
    assert decision.source == "system"
    assert QualityFlag.PARSE_ERROR in decision.flags


@hypothesis_or_cases
def test_low_cosine_always_uncertain(primary, audit, cosine, evidence):
    if cosine >= DEFAULT_CONFIG.retrieval_low_cosine_threshold:
        return
    inp = ClaimInput(primary, audit, cosine, evidence, parse_failed=False)
    decision = determine_decision(inp)
    assert decision.label == Label.RETRIEVAL_UNCERTAIN
    assert decision.source == "system"
    assert QualityFlag.RETRIEVAL_LOW_COSINE in decision.flags


@hypothesis_or_cases
def test_above_threshold_non_system_inputs_do_not_emit_system_labels(primary, audit, cosine, evidence):
    if cosine < DEFAULT_CONFIG.retrieval_low_cosine_threshold:
        return
    if primary in SYSTEM_LABELS or audit in SYSTEM_LABELS:
        return
    inp = ClaimInput(primary, audit, cosine, evidence, parse_failed=False)
    assert determine_decision(inp).label not in SYSTEM_LABELS


def test_supported_exact_unverified_downgrades_to_not_in_context():
    inp = ClaimInput(Label.SUPPORTED_EXACT, None, 0.8, False, False)
    decision = determine_decision(inp)
    assert decision.label == Label.NOT_IN_CONTEXT
    assert decision.source == "downgrade"
    assert QualityFlag.EVIDENCE_UNVERIFIED in decision.flags


def test_supported_paraphrase_unverified_downgrades_to_partially_supported():
    inp = ClaimInput(Label.SUPPORTED_PARAPHRASE, None, 0.8, False, False)
    decision = determine_decision(inp)
    assert decision.label == Label.PARTIALLY_SUPPORTED
    assert decision.source == "downgrade"


@pytest.mark.parametrize("label", [Label.PARTIALLY_SUPPORTED, Label.NOT_IN_CONTEXT, Label.CONTRADICTED])
def test_unverified_evidence_does_not_downgrade_non_supported_labels(label):
    inp = ClaimInput(label, None, 0.8, False, False)
    decision = determine_decision(inp)
    assert decision.label == label
    assert decision.source == "primary"


@pytest.mark.parametrize("evidence", [None, True])
def test_evidence_none_or_verified_does_not_downgrade(evidence):
    inp = ClaimInput(Label.SUPPORTED_EXACT, None, 0.8, evidence, False)
    assert determine_decision(inp).label == Label.SUPPORTED_EXACT


def test_audit_stricter_actually_overrides():
    inp = ClaimInput(Label.SUPPORTED_PARAPHRASE, Label.CONTRADICTED, 0.9, True, False)
    decision = determine_decision(inp)
    assert decision.label == Label.CONTRADICTED
    assert decision.source == "audit_override"
    assert QualityFlag.AUDIT_OVERRIDDEN in decision.flags


def test_audit_softer_never_overrides():
    inp = ClaimInput(Label.CONTRADICTED, Label.SUPPORTED_EXACT, 0.9, True, False)
    decision = determine_decision(inp)
    assert decision.label == Label.CONTRADICTED
    assert QualityFlag.AUDIT_DISAGREES_SOFTER in decision.flags


def test_audit_equal_no_override():
    inp = ClaimInput(Label.NOT_IN_CONTEXT, Label.NOT_IN_CONTEXT, 0.9, True, False)
    decision = determine_decision(inp)
    assert decision.label == Label.NOT_IN_CONTEXT
    assert decision.source == "primary"


@pytest.mark.parametrize("primary", [Label.PARSE_ERROR, Label.RETRIEVAL_UNCERTAIN])
def test_audit_never_overrides_primary_system_labels(primary):
    inp = ClaimInput(primary, Label.CONTRADICTED, 0.9, True, False)
    decision = determine_decision(inp)
    assert decision.label == primary
    assert decision.source == "system"
    assert QualityFlag.AUDIT_DISAGREES_WITH_SYSTEM in decision.flags


@pytest.mark.parametrize("audit", [Label.PARSE_ERROR, Label.RETRIEVAL_UNCERTAIN])
def test_audit_system_label_never_overrides_primary(audit):
    inp = ClaimInput(Label.SUPPORTED_EXACT, audit, 0.9, True, False)
    decision = determine_decision(inp)
    assert decision.label == Label.SUPPORTED_EXACT
    assert QualityFlag.AUDIT_DISAGREES_WITH_SYSTEM in decision.flags


@hypothesis_or_cases
def test_determine_decision_total_function(primary, audit, cosine, evidence):
    inp = ClaimInput(primary, audit, cosine, evidence, parse_failed=False)
    assert isinstance(determine_decision(inp), ClaimDecision)


def test_decisions_immutable():
    decision = determine_decision(ClaimInput(Label.SUPPORTED_EXACT, None, 0.8, True, False))
    with pytest.raises(FrozenInstanceError):
        decision.label = Label.CONTRADICTED


# -----------------------------------------------------------------------------
# Fix 1 regression tests — Evidence-Downgrade must NOT block Audit-Override
# -----------------------------------------------------------------------------


def test_evidence_downgrade_does_not_block_audit_override():
    """v4.0 Regression: supported_exact + evidence=False + audit=contradicted MUST → contradicted.

    Vorher hat rule_evidence_downgrade terminal returnt → not_in_context, Audit blockiert.
    Jetzt: Downgrade ist Mutator → Audit sieht not_in_context → contradicted ist strenger → wins.
    """
    inp = ClaimInput(Label.SUPPORTED_EXACT, Label.CONTRADICTED, 0.8, False, False)
    decision = determine_decision(inp)
    assert decision.label == Label.CONTRADICTED
    assert decision.source == "audit_override"
    # Downgrade-Flag bleibt erhalten, plus Override-Flag
    assert QualityFlag.AUDIT_OVERRIDDEN in decision.flags
    assert QualityFlag.EVIDENCE_UNVERIFIED in decision.flags


def test_evidence_downgrade_preserved_when_audit_softer():
    """Wenn Audit weicher als downgraded Label → Downgrade-Ergebnis bleibt."""
    inp = ClaimInput(Label.SUPPORTED_EXACT, Label.SUPPORTED_PARAPHRASE, 0.8, False, False)
    decision = determine_decision(inp)
    # downgrade → NOT_IN_CONTEXT (Rank 3), audit SUPPORTED_PARAPHRASE (Rank 1) ist weicher
    assert decision.label == Label.NOT_IN_CONTEXT
    assert QualityFlag.EVIDENCE_UNVERIFIED in decision.flags


# -----------------------------------------------------------------------------
# Fix 3 — ClaimInput-Validierung
# -----------------------------------------------------------------------------


def test_claim_input_rejects_nan_cosine():
    with pytest.raises(ValueError, match="finite"):
        ClaimInput(Label.SUPPORTED_EXACT, None, float("nan"), True, False)


def test_claim_input_rejects_inf_cosine():
    with pytest.raises(ValueError, match="finite"):
        ClaimInput(Label.SUPPORTED_EXACT, None, float("inf"), True, False)


def test_claim_input_rejects_out_of_range_cosine():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ClaimInput(Label.SUPPORTED_EXACT, None, 1.5, True, False)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ClaimInput(Label.SUPPORTED_EXACT, None, -0.1, True, False)


def test_claim_input_rejects_inconsistent_parse_failed_evidence():
    """parse_failed=True + evidence_verified gesetzt ist logisch inkonsistent."""
    with pytest.raises(ValueError, match="inconsistent"):
        ClaimInput(Label.SUPPORTED_EXACT, None, 0.8, True, True)
    with pytest.raises(ValueError, match="inconsistent"):
        ClaimInput(Label.SUPPORTED_EXACT, None, 0.8, False, True)


def test_claim_input_rejects_non_label_types():
    with pytest.raises(TypeError):
        ClaimInput("supported_exact", None, 0.8, True, False)  # str statt Label
    with pytest.raises(TypeError):
        ClaimInput(Label.SUPPORTED_EXACT, "contradicted", 0.8, True, False)


def test_claim_input_accepts_valid():
    # Valide Kombinationen — kein Raise
    ClaimInput(Label.SUPPORTED_EXACT, None, 0.0, None, False)
    ClaimInput(Label.SUPPORTED_EXACT, None, 1.0, True, False)
    ClaimInput(Label.SUPPORTED_EXACT, Label.CONTRADICTED, 0.5, False, False)
    ClaimInput(Label.SUPPORTED_EXACT, None, 0.5, None, True)  # parse_failed mit evidence=None ok
