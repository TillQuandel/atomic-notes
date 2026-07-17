"""#316: der Stage-8-Run-Ende-Print in orchestrator.main() baute `cov_rates` ueber
`r.get("coverage_factual", r.get("coverage_rate", -1.0)) >= 0` -- crasht mit
TypeError, sobald `coverage_factual` explizit als `None` gespeichert ist (der
Normalfall seit eval_version 4.x, #233): `dict.get`s Default greift nur bei
fehlendem Key, `r.get("coverage_factual", ...)` liefert dann `None`, und
`None >= 0` ist in Python 3 ein TypeError.

Fix: `_stage8_report_averages()` (extrahierte, pure Funktion) nutzt
`coverage_value()` (eval_common.py, #316-Helper), der `None` und fehlenden Key
gleich behandelt.

RED vor dem Fix: TypeError beim Aufruf mit einer Zeile, deren `coverage_factual`
explizit `None` ist (Bestandsmuster jeder aktuellen eval_version=4.x-Zeile).
"""

from __future__ import annotations

from generative.orchestrator import _stage8_report_averages


def test_stage8_report_averages_none_coverage_factual_does_not_crash():
    eval_results = [
        {"hallucination_rate": 0.1, "coverage_factual": None, "coverage_rate": 0.7},
        {"hallucination_rate": 0.3, "coverage_factual": None, "coverage_rate": 0.9},
    ]
    avg_hall, avg_cov = _stage8_report_averages(eval_results)
    assert avg_hall == 0.2
    assert avg_cov == 0.8  # (0.7 + 0.9) / 2, ueber coverage_rate-Fallback


def test_stage8_report_averages_real_zero_coverage_factual_not_swallowed():
    eval_results = [{"hallucination_rate": 0.1, "coverage_factual": 0.0, "coverage_rate": 0.9}]
    avg_hall, avg_cov = _stage8_report_averages(eval_results)
    assert avg_cov == 0.0  # nicht 0.9


def test_stage8_report_averages_empty_results_returns_none():
    assert _stage8_report_averages([]) == (None, None)


def test_stage8_report_averages_no_valid_hall_returns_none():
    eval_results = [{"hallucination_rate": -1.0, "coverage_factual": 0.5}]
    assert _stage8_report_averages(eval_results) == (None, None)
