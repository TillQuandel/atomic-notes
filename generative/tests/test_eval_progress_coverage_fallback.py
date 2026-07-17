"""#316: `eval_progress.py::print_table` berechnete `cov_f` bisher ueber
`r.get("coverage_factual", r.get("coverage_rate", 0))` -- bei einem explizit
gespeicherten `coverage_factual=None` (strukturell der Fall fuer JEDE
`eval_version=4.x`-Zeile, #233) faellt `dict.get` NICHT auf `coverage_rate`
zurueck (Default greift nur bei fehlendem Key), sondern liefert `None` --
`{None:>5.1%}` crasht mit TypeError.

RED vor dem Fix: TypeError beim Formatieren. GREEN: `coverage_value()`
(#316-Helper, `eval_common.py`) faellt korrekt auf `coverage_rate` zurueck.
"""

from __future__ import annotations

from generative.eval_progress import print_table


def _record(**overrides):
    base = {
        "version": "v1",
        "eval_version": "4.1",
        "language": "DE",
        "note": "n1",
        "timestamp": "2026-01-01T00:00",
        "anchors_confirmed": 1,
        "anchors_uncertain": 0,
        "anchors_hallucinated": 0,
        "hallucination_rate": 0.1,
        "coverage_factual": None,
        "coverage_rate": 0.7,
        "source_coverage": 0.0,
        "tokens_total": 100,
        "wall_time_s": 1.0,
        "pdf": "a.pdf",
    }
    base.update(overrides)
    return base


def test_print_table_coverage_factual_none_falls_back_to_coverage_rate(capsys):
    print_table([_record()])
    out = capsys.readouterr().out
    assert "70.0%" in out  # coverage_rate, nicht 0.0% (verschluckter Fallback)


def test_print_table_coverage_factual_real_zero_not_swallowed(capsys):
    print_table([_record(coverage_factual=0.0, coverage_rate=0.9)])
    out = capsys.readouterr().out
    assert "0.0%" in out
    assert "90.0%" not in out
