"""Till-Befund (2026-07-14): die KPI-Gruppen "Qualitaet" und "Kosten &
Performance" gehoeren zur selben Pipeline-Version, aber nur der "Qualitaet"-
Header zeigte das Versions-Label (id="kpi-ver-label"). Der "Kosten &
Performance"-Header hatte gar keins -> User konnte nicht erkennen, dass beide
Gruppen zur gleichen Version gehoeren.

Fix: zweites Label (id="kpi-ver-label-perf") am "Kosten & Performance"-Header,
von DERSELBEN JS-Zuweisung befuellt wie das bestehende "Qualitaet"-Label (SSoT
— ein Wert, zwei Anzeigeorte statt zweiter Datenquelle).
"""

from __future__ import annotations

from generative.eval_dashboard_server import _build_live_html


def test_cost_perf_header_has_version_label_span():
    html = _build_live_html()
    assert 'id="kpi-ver-label-perf"' in html
    # Label sitzt im "Kosten & Performance"-Sub-Head, nicht irgendwo sonst im Dokument.
    perf_head_start = html.index('<span class="sub-label">Kosten &amp; Performance')
    perf_head_end = html.index("</div>", perf_head_start)
    assert 'id="kpi-ver-label-perf"' in html[perf_head_start:perf_head_end]


def test_both_kpi_group_version_labels_set_from_same_js_assignment():
    html = _build_live_html()
    # Beide Labels muessen im selben renderData()-Block gesetzt werden, damit
    # sie nie auseinanderlaufen (SSoT: eine Zuweisung von d.kpis.kpi_version).
    js_start = html.index("const verLbl = document.getElementById('kpi-ver-label');")
    js_snippet = html[js_start : js_start + 400]
    assert "kpi-ver-label-perf" in js_snippet
    assert js_snippet.count("d.kpis.kpi_version") == 2
