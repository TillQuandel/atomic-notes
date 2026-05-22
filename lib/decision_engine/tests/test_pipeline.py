from __future__ import annotations

from decision_engine import ClaimInput, Label, determine_decision
from decision_engine.models import QualityFlag


def test_parse_error_beats_low_cosine_and_audit():
    # parse_failed=True erfordert evidence_verified=None (Validation, siehe ClaimInput.__post_init__).
    inp = ClaimInput(Label.SUPPORTED_EXACT, Label.CONTRADICTED, 0.0, None, True)
    decision = determine_decision(inp)
    assert decision.label == Label.PARSE_ERROR
    assert decision.flags == frozenset({QualityFlag.PARSE_ERROR})


def test_low_cosine_beats_evidence_downgrade_and_audit():
    inp = ClaimInput(Label.SUPPORTED_EXACT, Label.CONTRADICTED, 0.1, False, False)
    decision = determine_decision(inp)
    assert decision.label == Label.RETRIEVAL_UNCERTAIN
    assert decision.flags == frozenset({QualityFlag.LOW_COSINE})


def test_evidence_downgrade_does_not_block_stricter_audit():
    # v4.1 Verhalten: Evidence-Downgrade ist Mutator, danach kann Audit überschreiben.
    # Vorher (v4.0): bleibt bei NOT_IN_CONTEXT — das war die Regression zu v3.2.
    inp = ClaimInput(Label.SUPPORTED_EXACT, Label.CONTRADICTED, 0.9, False, False)
    decision = determine_decision(inp)
    assert decision.label == Label.CONTRADICTED
    assert decision.source == "audit_override"
    assert QualityFlag.EVIDENCE_UNVERIFIED in decision.flags  # Downgrade-Flag bleibt erhalten
    assert QualityFlag.AUDIT_OVERRODE in decision.flags


def test_evidence_downgrade_persists_when_audit_softer():
    # Audit ist weicher als downgraded Label → Downgrade-Ergebnis bleibt.
    inp = ClaimInput(Label.SUPPORTED_EXACT, Label.SUPPORTED_PARAPHRASE, 0.9, False, False)
    decision = determine_decision(inp)
    assert decision.label == Label.NOT_IN_CONTEXT  # downgrade, Audit zu weich
    assert QualityFlag.EVIDENCE_UNVERIFIED in decision.flags


def test_default_primary_preserves_supported_claim():
    inp = ClaimInput(Label.SUPPORTED_PARAPHRASE, None, 0.9, True, False)
    decision = determine_decision(inp)
    assert decision.label == Label.SUPPORTED_PARAPHRASE
    assert decision.flags == frozenset()
    assert decision.source == "primary"
