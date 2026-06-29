"""Decision-Pipeline: orchestriert die Rules in 3 Phasen.

# CLAUDE-PATTERN: 3-Phasen-Trennung (Terminal → Mutator → Decider) löst die v4-Regression:
# Wenn Evidence-Downgrade als terminale Rule lief, konnte Audit-Override danach nicht
# mehr greifen. Jetzt ist Downgrade eine Input-Mutation, danach läuft Audit normal.
"""

from __future__ import annotations

from collections.abc import Callable

from decision_engine.models import ClaimDecision, ClaimInput
from decision_engine.rules import (
    DEFAULT_CONFIG,
    RulesConfig,
    apply_evidence_downgrade,
    rule_audit_stricter_override,
    rule_default_primary,
    rule_low_cosine_system,
    rule_parse_error,
)


TerminalRule = Callable[[ClaimInput, RulesConfig], ClaimDecision | None]

# Phase 1: System-Overrides — höchste Priorität, sofortiges Pipeline-Ende.
TERMINAL_RULES: tuple[TerminalRule, ...] = (
    rule_parse_error,
    rule_low_cosine_system,
)


def determine_decision(inp: ClaimInput, config: RulesConfig = DEFAULT_CONFIG) -> ClaimDecision:
    """Determiniert finale Decision in 3 Phasen.

    Phase 1: Terminal-Rules (parse_error, low_cosine) — bei Match Ende
    Phase 2: Mutator (evidence_downgrade) — modifiziert Input, gibt Flags
    Phase 3: Audit-Override oder Default — finale Decision auf bereits gedowngradetem Input
    """
    # Phase 1: Terminal-Overrides
    for rule in TERMINAL_RULES:
        decision = rule(inp, config)
        if decision is not None:
            return decision

    # Phase 2: Evidence-Downgrade (Mutation)
    mutated_inp, downgrade_flags = apply_evidence_downgrade(inp, config)

    # Phase 3: Audit-Override
    audit_decision = rule_audit_stricter_override(mutated_inp, config)
    if audit_decision is not None:
        # Audit-Decision: hänge Downgrade-Flags an falls vorhanden
        if downgrade_flags:
            merged_flags = audit_decision.flags | downgrade_flags
            return ClaimDecision(label=audit_decision.label, flags=merged_flags, source=audit_decision.source)
        return audit_decision

    # Fallback: Default mit Downgrade-Flags
    default = rule_default_primary(mutated_inp, config)
    if downgrade_flags:
        merged_flags = default.flags | downgrade_flags
        # Kanonischer Source-Wert laut models.py-Contract: "downgrade" (#32)
        return ClaimDecision(
            label=default.label, flags=merged_flags, source="downgrade" if downgrade_flags else default.source
        )
    return default
