"""Aggregation der Decisions zu Metriken.

# CLAUDE-PATTERN: Alle Rates verhalten sich konsistent — bei total=0 oder
# valid_claims=0 → Metric.invalid(). Sentinel-Wert -1.0 ist universell, kein
# Mix aus 0.0 und -1.0 mehr (Fix für v4-Aggregation-Inkonsistenz).
"""

from __future__ import annotations

from collections import Counter

from decision_engine.models import SYSTEM_LABELS, ClaimDecision, Label, Metric
from decision_engine.rules import metric_count_key


def _valid(decisions: list[ClaimDecision]) -> list[ClaimDecision]:
    return [decision for decision in decisions if decision.label not in SYSTEM_LABELS]


def _rate(numerator: int, denominator: int) -> Metric:
    """Konsistente Rate-Berechnung. Bei denominator=0 → Metric.invalid()."""
    if denominator == 0:
        return Metric.invalid()
    return Metric(round(numerator / denominator, 3), True)


def hallucination_rate(decisions: list[ClaimDecision]) -> Metric:
    """Anteil hallucinated (NOT_IN_CONTEXT + CONTRADICTED) an valid_claims."""
    valid = _valid(decisions)
    hallucinated = sum(1 for d in valid if d.label in {Label.NOT_IN_CONTEXT, Label.CONTRADICTED})
    return _rate(hallucinated, len(valid))


def confirmed_rate(decisions: list[ClaimDecision]) -> Metric:
    """Anteil confirmed (SUPPORTED_EXACT + SUPPORTED_PARAPHRASE) an valid_claims."""
    valid = _valid(decisions)
    confirmed = sum(1 for d in valid if d.label in {Label.SUPPORTED_EXACT, Label.SUPPORTED_PARAPHRASE})
    return _rate(confirmed, len(valid))


def partial_rate(decisions: list[ClaimDecision]) -> Metric:
    """Anteil PARTIALLY_SUPPORTED an valid_claims (konsistent mit hallucination/confirmed)."""
    valid = _valid(decisions)
    partial = sum(1 for d in valid if d.label is Label.PARTIALLY_SUPPORTED)
    return _rate(partial, len(valid))


def uncertain_rate(decisions: list[ClaimDecision]) -> Metric:
    """Anteil RETRIEVAL_UNCERTAIN an total. System-Metrik, nicht über valid_claims.

    # CLAUDE-PATTERN: uncertain/parse_error sind System-Metriken — Nenner ist
    # total (nicht valid_claims), weil sie selbst die Definition von valid bestimmen.
    """
    total = len(decisions)
    uncertain = sum(1 for d in decisions if d.label is Label.RETRIEVAL_UNCERTAIN)
    return _rate(uncertain, total)


def parse_error_rate(decisions: list[ClaimDecision]) -> Metric:
    """Anteil PARSE_ERROR an total."""
    total = len(decisions)
    errors = sum(1 for d in decisions if d.label is Label.PARSE_ERROR)
    return _rate(errors, total)


def label_counts(decisions: list[ClaimDecision]) -> dict[Label, int]:
    counts = Counter(decision.label for decision in decisions)
    return {label: counts[label] for label in Label}


def _metric_dict(metric: Metric) -> dict[str, object]:
    """Metric als {value, valid} Dict (statt nur value + separates rate_valid)."""
    return {"value": metric.value, "valid": metric.valid}


def aggregate(decisions: list[ClaimDecision]) -> dict[str, object]:
    """JSON-ready Aggregation. Alle Rates als nested {value, valid} Dict.

    # CLAUDE-PATTERN: Nested Dict-Struktur ersetzt Sentinel + separates `rate_valid`-Feld.
    # Jede Metrik trägt eigene Validity-Information, kein irreführendes globales Flag.
    # Backward-Compat-Felder (anchors_*, hallucination_rate-Flach) bleiben für Dashboard.
    """
    counts = label_counts(decisions)
    valid = _valid(decisions)
    total = len(decisions)
    confirmed = counts[Label.SUPPORTED_EXACT] + counts[Label.SUPPORTED_PARAPHRASE]
    hallucinated = counts[Label.NOT_IN_CONTEXT] + counts[Label.CONTRADICTED]

    hall_metric = hallucination_rate(decisions)
    conf_metric = confirmed_rate(decisions)
    part_metric = partial_rate(decisions)
    unc_metric = uncertain_rate(decisions)
    parse_metric = parse_error_rate(decisions)

    # claim_support_rate = (confirmed + partially_supported) / valid_claims
    support_metric = _rate(confirmed + counts[Label.PARTIALLY_SUPPORTED], len(valid))

    result: dict[str, object] = {
        # Counts
        "claims_total": total,
        "valid_claims": len(valid),
        "anchors_total": total,
        "anchors_confirmed": confirmed,
        "anchors_hallucinated": hallucinated,
        # Strukturierte Metriken (jede mit eigener validity)
        "metrics": {
            "hallucination_rate": _metric_dict(hall_metric),
            "confirmed_rate": _metric_dict(conf_metric),
            "partial_rate": _metric_dict(part_metric),
            "uncertain_rate": _metric_dict(unc_metric),
            "parse_error_rate": _metric_dict(parse_metric),
            "claim_support_rate": _metric_dict(support_metric),
        },
        # Backward-Compat-Aliases (flache Sentinel-Werte für Dashboard/Orchestrator)
        "hallucination_rate": hall_metric.value,
        "confirmed_rate": conf_metric.value,
        "claim_support_rate": support_metric.value,
        "partial_rate": part_metric.value,
        "uncertain_rate": unc_metric.value,
        # CLAUDE-PATTERN: retrieval_failure_rate ist semantisch identisch zu uncertain_rate,
        # bleibt als Alias für Konsumenten die den alten Key kennen (z.B. eval_quality_v4).
        "retrieval_failure_rate": unc_metric.value,
        "parse_error_rate": parse_metric.value,
    }
    # Per-Label Counts
    for label, count in counts.items():
        result[f"claims_{metric_count_key(label)}"] = count
    return result
