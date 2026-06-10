"""Tests für die avg_agree-Aggregation im Eval-Dashboard.

Review-Finding (#24-Branch): Bei vorhandenen Rows, deren agreement aber komplett
None ist (z.B. quality_history ohne AGENT_VERSION-Einträge → collect.py schreibt
NULL), ergab `sum(...)/max(1, count)` 0.0 — das Frontend zeigte "0 %"
(katastrophale Fehlkalibrierung) statt "–" (keine Daten).
"""
from __future__ import annotations
import sys
from pathlib import Path


from generative.eval_dashboard_server import _avg_agreement


def test_average_over_non_none_values():
    rows = [{"agreement": 80.0}, {"agreement": 60.0}, {"agreement": None}]
    assert _avg_agreement(rows) == 70.0


def test_empty_rows_returns_none():
    assert _avg_agreement([]) is None


def test_rows_with_all_none_agreement_returns_none():
    # Der Bug-Fall: Rows existieren, aber kein agreement-Wert → None, nicht 0.0.
    rows = [{"agreement": None}, {"agreement": None}]
    assert _avg_agreement(rows) is None
