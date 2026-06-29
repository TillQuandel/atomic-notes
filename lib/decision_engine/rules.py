"""Decision-Rules für die Pipeline.

Jede Rule ist eine reine Funktion `(ClaimInput, RulesConfig) -> Optional[ClaimDecision]`.
Rules sind in zwei Klassen geteilt:
- TERMINAL_RULES: Erste Match-Decision wird returnt, Pipeline endet (System-Overrides).
- MUTATOR: apply_evidence_downgrade modifiziert ClaimInput, returnt nicht (siehe pipeline.py).

# CLAUDE-PATTERN: Evidence-Downgrade ist Mutator, kein Decider. Frühere v4-Version
# returnte hier ein ClaimDecision was Audit-Override blockierte → Regression zu v3.2.
# Fix: Downgrade modifiziert nur das primary_label, danach kann Audit normal laufen.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import assert_never

from decision_engine.models import (
    STRICTNESS_RANK,
    SYSTEM_LABELS,
    ClaimDecision,
    ClaimInput,
    Label,
    QualityFlag,
)


@dataclass(frozen=True)
class RulesConfig:
    """Konfiguration der Rules. Single source of truth für Thresholds.

    # CLAUDE-PATTERN: RulesConfig ist autoritativ — Caller (z.B. eval_quality_v4)
    # importieren den Wert hier, statt eine eigene Konstante zu pflegen. Verhindert Drift.
    """

    retrieval_low_cosine_threshold: float = 0.4


DEFAULT_CONFIG = RulesConfig()


def _decision(label: Label, flags: set[QualityFlag] | frozenset[QualityFlag] | None, source: str) -> ClaimDecision:
    return ClaimDecision(label=label, flags=frozenset(flags or ()), source=source)


def _source_for(label: Label) -> str:
    return "system" if label in SYSTEM_LABELS else "primary"


# ---------------------------------------------------------------------------
# TERMINAL RULES — bei Match Pipeline-Ende
# ---------------------------------------------------------------------------


def rule_parse_error(inp: ClaimInput, config: RulesConfig = DEFAULT_CONFIG) -> ClaimDecision | None:
    """Höchste Priorität: parse_failed → PARSE_ERROR, immer terminal."""
    if inp.parse_failed:
        return _decision(Label.PARSE_ERROR, {QualityFlag.PARSE_ERROR}, "system")
    return None


def rule_low_cosine_system(inp: ClaimInput, config: RulesConfig = DEFAULT_CONFIG) -> ClaimDecision | None:
    """Zweite Priorität: cosine unter Threshold → RETRIEVAL_UNCERTAIN, immer terminal.

    Cosine ist garantiert finite und in [0,1] (siehe ClaimInput.__post_init__).
    """
    if inp.cosine < config.retrieval_low_cosine_threshold:
        return _decision(Label.RETRIEVAL_UNCERTAIN, {QualityFlag.RETRIEVAL_LOW_COSINE}, "system")
    return None


# ---------------------------------------------------------------------------
# MUTATOR — modifiziert ClaimInput, kein Pipeline-Ende
# ---------------------------------------------------------------------------


def apply_evidence_downgrade(
    inp: ClaimInput, config: RulesConfig = DEFAULT_CONFIG
) -> tuple[ClaimInput, frozenset[QualityFlag]]:
    """Downgrade supported_* Labels wenn Evidence nicht verifiziert.

    Returnt modifizierten ClaimInput + Flags zur Aufnahme in finale Decision.
    # CLAUDE-PATTERN: Mutator-Pattern — gibt geänderten Input zurück statt Decision
    # zu finalisieren. Damit Audit-Rule danach laufen kann (Fix für v4-Regression).
    """
    if inp.evidence_verified is not False:
        return inp, frozenset()

    if inp.primary_label is Label.SUPPORTED_EXACT:
        return replace(inp, primary_label=Label.NOT_IN_CONTEXT), frozenset(
            {QualityFlag.EVIDENCE_UNVERIFIED, QualityFlag.EVIDENCE_FABRICATED}
        )
    if inp.primary_label is Label.SUPPORTED_PARAPHRASE:
        return replace(inp, primary_label=Label.PARTIALLY_SUPPORTED), frozenset({QualityFlag.EVIDENCE_UNVERIFIED})
    return inp, frozenset()


# ---------------------------------------------------------------------------
# DECIDER — finale Entscheidung wenn nichts vorher terminal war
# ---------------------------------------------------------------------------


def rule_audit_stricter_override(inp: ClaimInput, config: RulesConfig = DEFAULT_CONFIG) -> ClaimDecision | None:
    """Audit-Override nur bei strengerem Label. Nie für/von Systemlabels."""
    audit = inp.audit_label
    primary = inp.primary_label
    if audit is None or audit is primary:
        return None
    if primary in SYSTEM_LABELS or audit in SYSTEM_LABELS:
        return _decision(primary, {QualityFlag.AUDIT_DISAGREES_WITH_SYSTEM}, _source_for(primary))

    primary_rank = STRICTNESS_RANK[primary]
    audit_rank = STRICTNESS_RANK[audit]
    if audit_rank > primary_rank:
        return _decision(audit, {QualityFlag.JUDGE_UNEINIG, QualityFlag.AUDIT_OVERRIDDEN}, "audit_override")
    return _decision(primary, {QualityFlag.AUDIT_DISAGREES_SOFTER}, "primary")


def rule_default_primary(inp: ClaimInput, config: RulesConfig = DEFAULT_CONFIG) -> ClaimDecision:
    """Catch-all: Primary-Label durchreichen wenn nichts gefeuert hat."""
    return _decision(inp.primary_label, None, _source_for(inp.primary_label))


# ---------------------------------------------------------------------------
# Label-Key-Mapping für Aggregation (mit assert_never für neue Labels)
# ---------------------------------------------------------------------------


def metric_count_key(label: Label) -> str:
    """Mapped Label-Enum auf JSON-Key. assert_never erzwingt Vollständigkeit.

    # CLAUDE-PATTERN: assert_never im match-Block macht neue Labels zu mypy/pyright-Fehlern.
    # Ohne diesen Catch-All würde ein neues Label still einen leeren String oder Crash erzeugen.
    """
    match label:
        case Label.SUPPORTED_EXACT:
            return "supported_exact"
        case Label.SUPPORTED_PARAPHRASE:
            return "supported_paraphrase"
        case Label.PARTIALLY_SUPPORTED:
            return "partially_supported"
        case Label.NOT_IN_CONTEXT:
            return "not_in_context"
        case Label.CONTRADICTED:
            return "contradicted"
        case Label.PARSE_ERROR:
            return "parse_error"
        case Label.RETRIEVAL_UNCERTAIN:
            return "retrieval_or_parse_uncertain"
        case _ as unreachable:
            assert_never(unreachable)
