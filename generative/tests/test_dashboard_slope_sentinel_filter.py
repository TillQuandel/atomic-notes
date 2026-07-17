"""#315: `_build_quality_chart_data` filterte hallucination_rate/coverage_rate beim
Aufbau der Slope-Datasets nur auf `is not None`, nicht auf den `-1.0`-Sentinel, den
`eval_quality_v4.py` bei ungueltigen Laeufen schreibt (z.B. `valid_claims == 0`).
Ein Sentinel `-1.0` wurde nach der `*100`-Rundung zu `-100.0` und rutschte als
valider Wert in die Slope-Mediane des Legacy-Einmal-Render-Pfads (`main()`/
`_build_html`). Fix: derselbe `>= 0`-Guard wie in `_chart_scatter`/
`_chart_scatter_versioned`.

RED vor dem Fix: `rows_clean[0]["hall"]`/`["cov"]` == -100.0 (Sentinel eingesickert).
GREEN: beide `None` (wie bei einer fehlenden Rate) -- der Slope-Median ignoriert
die Zeile bereits ueber den bestehenden `is not None`-Filter (Zeilen 1878/1880).
"""

from __future__ import annotations

from generative.eval_dashboard import _build_quality_chart_data


def test_build_quality_chart_data_hall_sentinel_becomes_none():
    rows = [
        {
            "hallucination_rate": -1.0,
            "coverage_factual": 0.5,
            "note": "n1",
            "pdf": "a.pdf",
            "version": "v1",
        }
    ]
    out = _build_quality_chart_data(rows)
    assert out["rows"][0]["hall"] is None  # nicht -100.0


def test_build_quality_chart_data_cov_sentinel_becomes_none():
    rows = [
        {
            "hallucination_rate": 0.1,
            "coverage_factual": -1.0,
            "coverage_rate": -1.0,
            "note": "n1",
            "pdf": "a.pdf",
            "version": "v1",
        }
    ]
    out = _build_quality_chart_data(rows)
    assert out["rows"][0]["cov"] is None  # nicht -100.0


def test_build_quality_chart_data_sentinel_excluded_from_slope_median():
    rows = [
        {
            "hallucination_rate": -1.0,
            "coverage_factual": 0.5,
            "note": "n1",
            "pdf": "a.pdf",
            "version": "v1",
        },
        {
            "hallucination_rate": 0.2,
            "coverage_factual": 0.5,
            "note": "n2",
            "pdf": "a.pdf",
            "version": "v1",
        },
    ]
    out = _build_quality_chart_data(rows)
    slope = out["slope_datasets"][0]
    # Median nur ueber die gueltige Zeile (20.0), nicht (-100.0 + 20.0)/2 = -40.0
    assert slope["hall_data"] == [20.0]
