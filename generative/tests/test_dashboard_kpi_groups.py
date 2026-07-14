"""Till-Entscheid (Nachfolge-PR zu #266): KPI-Kacheln des Eval-Dashboards in
zwei beschriftete Gruppen aufgeteilt — "Qualität" (Automatisch akzeptiert,
Fehlerquote, Belegrate, Evaluierte Notes) und "Kosten & Performance"
(Agent-Rechenzeit, Wall-Clock, Billable Tokens, Tokens ohne Cache,
API-Kosten). Explizites Anti-Pattern laut Till: KEIN Slider/Karussell.

Diese Tests decken nur das ausgelieferte HTML/JS (String-Assertions) ab —
die eigentliche DOM-Aufteilung passiert client-seitig in renderKpis() per
Array-Slice über die (unveraendert) IDs/Labels/Tooltips. Ein Browser-Rendering-
Test folgt separat per Playwright-Sichtpruefung (siehe PR-Beschreibung).
"""

from __future__ import annotations

from generative.eval_dashboard_server import _build_live_html


def test_html_has_two_labeled_kpi_group_containers():
    html = _build_live_html()
    assert 'id="kpis-quality"' in html
    assert 'id="kpis-perf"' in html
    # Gruppen-Label "Kosten & Performance" existiert im Dashboard bereits als
    # Sektions-Name (id="s-cost") — hier dieselbe Bezeichnung fuer die neue
    # Sub-Head wiederverwendet (SSoT-Konsistenz statt neuer Begriff).
    assert "Kosten &amp; Performance" in html


def test_html_has_no_slider_or_carousel_pattern():
    html = _build_live_html()
    lowered = html.lower()
    assert "carousel" not in lowered
    assert "swiper" not in lowered
    assert "slick" not in lowered


def test_kpi_group_containers_appear_in_document_order_quality_then_perf():
    html = _build_live_html()
    quality_idx = html.index('id="kpis-quality"')
    perf_idx = html.index('id="kpis-perf"')
    assert quality_idx < perf_idx


def test_kpi_defs_order_matches_group_split_quality_first_four_perf_last_five():
    """Die Gruppierung wird client-seitig per Array-Slice(0,4)/Slice(4) über
    die bestehende kpiDefs-Reihenfolge gebaut — dieser Test verankert, dass
    genau die 4 Qualitäts-Kacheln zuerst und die 5 Kosten/Performance-Kacheln
    danach im Array stehen (Slice-Grenze bei Index 4). Bricht laut, falls die
    Reihenfolge oder Anzahl je Gruppe sich künftig unbemerkt verschiebt.
    """
    html = _build_live_html()
    quality_labels = [
        "label:'Automatisch akzeptiert'",
        "label:'Fehlerquote'",
        "label:'Belegrate'",
        "label:'Evaluierte Notes'",
    ]
    perf_labels = [
        "label:'Agent-Rechenzeit (Summe)'",
        "label:'Wall-Clock (aktuell)'",
        "label:'Billable Tokens'",
        "label:'Tokens (ohne Cache)'",
        "label:'API-Kosten',",
    ]
    positions = [html.index(label) for label in quality_labels + perf_labels]
    assert positions == sorted(positions), (
        "kpiDefs-Reihenfolge muss alle 4 Qualitaets-Labels VOR allen 5 "
        "Kosten&Performance-Labels fuehren (Slice-Split-Voraussetzung)."
    )
    # Slice-Grenze bei Index 4 im tatsaechlichen Render-Code verankert.
    assert ".slice(0, 4)" in html
    assert ".slice(4)" in html


def test_spark_row_relocates_to_clicked_kpi_group_on_open():
    """#274-Nachbesserung (Till-Live-Fund): die EINE #spark-row saß nach dem
    Gruppen-Umbau (#274) fix im HTML nach #kpis-perf — ein Klick auf eine
    Qualitäts-Kachel (idx<4) öffnete die Trend-Row also unter der FALSCHEN
    (unteren) Gruppe statt direkt unter der eigenen. Fix: _openKpiSpark hängt
    die Row per insertAdjacentElement('afterend', ...) dynamisch unter die
    Gruppe der angeklickten Kachel — dieselbe idx<4-Grenze wie der
    kpiDefs-Slice-Split oben.
    """
    html = _build_live_html()
    open_fn = html.split("window._openKpiSpark = function(idx) {")[1].split("window.closeKpiSpark = function()")[0]
    assert "insertAdjacentElement('afterend', row)" in open_fn
    assert "kpis-quality" in open_fn and "kpis-perf" in open_fn
    assert "idx < 4" in open_fn
