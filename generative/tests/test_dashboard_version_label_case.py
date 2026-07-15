"""Test für #U7 (Multi-Perspektiven-Review 2026-07-15): kpi-ver-label wird per
CSS uppercase gerendert.

Befund: `.sub-label` hat `text-transform: uppercase` (Sub-Head-Beschriftungen
wie "QUALITÄT" sollen groß sein). Das trifft aber auch die eingebettete
Versionsnummer `#kpi-ver-label` ("V0.3.143" statt "v0.3.143") -- Pipeline-
Versionen sind kleingeschrieben (`v0.3.143`), Großschreibung verfälscht sie.
Fix: `text-transform: none` auf dem Versions-Span, damit `.sub-label`s
Uppercase-Regel es nicht mehr trifft.
"""

from __future__ import annotations

from generative.eval_dashboard_server import _build_live_html


def test_kpi_ver_label_excluded_from_uppercase_transform():
    html = _build_live_html()
    assert "#kpi-ver-label" in html
    assert "text-transform: none" in html


def test_kpi_ver_label_perf_also_excluded_for_future_pr291_label():
    # #291 (offen, Stand dieser Änderung) fügt ein zweites Versions-Label
    # id="kpi-ver-label-perf" am "Kosten & Performance"-Header hinzu -- der
    # CSS-Selektor deckt es bereits ab, damit es denselben Fix erbt sobald
    # #291 gemergt ist (kein Nacharbeits-Bedarf).
    html = _build_live_html()
    assert "#kpi-ver-label-perf" in html
