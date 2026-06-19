"""Tests für die Dashboard-Follow-ups (atomic-notes Issue #36, 2026-06-19).

- P4: version_delta — Delta neueste-vs-Vorversion pro KPI mit N-Guard.
- P2: scaling-Recency-Flag — alte Versions-Ären markieren (dimmen statt mischen).
- P1: _chart_longitudinal Median-über-PDFs-Serie (Anti-Spaghetti).
"""
from __future__ import annotations

import pytest

from generative.eval_dashboard import version_delta, mark_scaling_recency


# ---------------------------------------------------------------- P4 fixtures
def _kpi_trend(**over):
    """sorted_pipeline_versions aufsteigend → neueste = letzte Position."""
    base = {
        "versions": ["v0.3.134", "v0.3.135"],
        "hall":     [12.0, 9.7],
        "cov":      [30.0, 35.0],
        "n":        [25, 22],
        "accept":   [50.0, 60.0],
    }
    base.update(over)
    return base


# ------------------------------------------------------------------- P4 tests
def test_version_delta_latest_prev_and_signed_delta():
    d = version_delta(_kpi_trend(), "hall")
    assert d["latest"] == 9.7
    assert d["prev"] == 12.0
    assert d["delta"] == pytest.approx(-2.3)   # Halluzination gesunken → negativ


def test_version_delta_positive_sign_for_rising_coverage():
    d = version_delta(_kpi_trend(), "cov")
    assert d["delta"] == pytest.approx(5.0)     # Coverage gestiegen → positiv


def test_version_delta_reliable_only_when_both_n_at_least_20():
    assert version_delta(_kpi_trend(n=[25, 22]), "hall")["reliable"] is True
    assert version_delta(_kpi_trend(n=[25, 5]), "hall")["reliable"] is False   # latest zu klein
    assert version_delta(_kpi_trend(n=[5, 25]), "hall")["reliable"] is False   # prev zu klein


def test_version_delta_no_previous_version_yields_none_delta():
    d = version_delta(
        {"versions": ["v0.3.135"], "hall": [9.7], "n": [22]}, "hall"
    )
    assert d["latest"] == 9.7
    assert d["prev"] is None
    assert d["delta"] is None
    assert d["reliable"] is False


def test_version_delta_none_metric_value_yields_none_delta():
    d = version_delta(_kpi_trend(hall=[12.0, None]), "hall")
    assert d["latest"] is None
    assert d["delta"] is None
    assert d["reliable"] is False


def test_version_delta_empty_trend_yields_none_delta():
    d = version_delta({"versions": [], "hall": [], "n": []}, "hall")
    assert d["latest"] is None
    assert d["prev"] is None
    assert d["delta"] is None
    assert d["reliable"] is False


# ------------------------------------------------------------------- P2 tests
def _pt(ver, key="a", x=1000, y=4):
    return {"x": x, "y": y, "key": key, "label": key, "ver": ver}


def test_mark_scaling_recency_flags_only_youngest_keep_versions():
    pts = [_pt(f"v0.3.{i}") for i in range(1, 13)]   # 12 Versionen, numerisch sortiert
    recent = {p["ver"]: p["recent"] for p in mark_scaling_recency(pts, keep=10)}
    assert recent["v0.3.1"] is False    # die zwei ältesten Versionen gedimmt
    assert recent["v0.3.2"] is False
    assert recent["v0.3.3"] is True
    assert recent["v0.3.12"] is True


def test_mark_scaling_recency_all_recent_when_fewer_than_keep():
    out = mark_scaling_recency([_pt("v0.1.0"), _pt("v0.2.0")], keep=10)
    assert all(p["recent"] for p in out)


def test_mark_scaling_recency_multiple_points_per_version():
    pts = [_pt("v0.1.0", key="a"), _pt("v0.1.0", key="b"), _pt("v0.9.0", key="a")]
    out = mark_scaling_recency(pts, keep=1)
    assert [p["recent"] for p in out] == [False, False, True]   # nur jüngste Version recent


def test_mark_scaling_recency_missing_version_not_recent():
    assert mark_scaling_recency([_pt(None)], keep=10)[0]["recent"] is False


def test_mark_scaling_recency_does_not_mutate_input():
    pts = [_pt("v0.1.0")]
    mark_scaling_recency(pts, keep=10)
    assert "recent" not in pts[0]
