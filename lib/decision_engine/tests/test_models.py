"""#318: historische eval_version=4.1-Zeilen tragen `decision_source="audit"`
(91 Claim-Vorkommen in 67 JSONL-Zeilen, Eval-Doku-Audit #313) -- aktueller Code
schreibt nur noch `"audit_override"` (rules.py::rule_audit_stricter_override).
Kein Code-Pfad schreibt den alten Wert mehr; Leser die nach `decision_source`
filtern/gruppieren muessen aber beide Werte kennen, sonst werden historische
Audit-Overrides beim Filtern auf "audit_override" stillschweigend unterschlagen.

`normalize_decision_source()` normalisiert einen GELESENEN Wert auf das
aktuelle Vokabular -- KEINE Mutation der Bestandsdaten (JSONL/DB bleiben
unveraendert), nur eine Lese-Seiten-Normalisierung.
"""

from __future__ import annotations

from decision_engine.models import normalize_decision_source


def test_normalize_decision_source_maps_legacy_audit_alias():
    assert normalize_decision_source("audit") == "audit_override"


def test_normalize_decision_source_current_vocabulary_passthrough():
    for value in ("primary", "audit_override", "system", "downgrade"):
        assert normalize_decision_source(value) == value


def test_normalize_decision_source_unknown_value_passthrough():
    # Kein stiller Datenverlust bei unbekannten Werten -- unveraendert durchreichen.
    assert normalize_decision_source("future_value") == "future_value"
