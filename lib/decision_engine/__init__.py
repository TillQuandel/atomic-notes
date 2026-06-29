"""decision_engine — Rule-Pipeline für Claim-Decisions.

Public API: determine_decision, Label, ClaimInput, ClaimDecision, Metric.
"""

from __future__ import annotations

from decision_engine.models import ClaimDecision, ClaimInput, Label, Metric, QualityFlag
from decision_engine.pipeline import determine_decision
from decision_engine.rules import RulesConfig, DEFAULT_CONFIG


__all__ = [
    "ClaimDecision",
    "ClaimInput",
    "DEFAULT_CONFIG",
    "Label",
    "Metric",
    "QualityFlag",
    "RulesConfig",
    "determine_decision",
]
