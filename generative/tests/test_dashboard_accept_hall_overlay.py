"""Punkt 3 (S2-C3, Multi-Perspektiven-Dashboard-Review 2026-07-15):
Accept×Hall-Overlay in der bestehenden Versions-Trend-Ansicht.

Befund: note_evals enthält nur AKZEPTIERTE Notes (Selektions-Bias) -- sinkt
die Akzeptanzquote gleichzeitig mit einer Fehlerquoten-Änderung, ist
Letztere teils Stichproben-Verschiebung statt reiner Qualitätseffekt, ohne
dass das im bestehenden Fehlerquote-Spark-Chart sichtbar wäre. Fix: die
Akzeptanzquote als zweite (gestrichelte, gedämpfte) Serie im BESTEHENDEN
Fehlerquote-Spark-Chart (_trendChart, Klick auf die Fehlerquote-KPI-Kachel)
-- kein neues Chart (Spec-Vorgabe).

Nachbesserung (Statistiker MEDIUM + UX-P3, Reviews 15./16.07 zu PR #312):
die gemeinsame 0-100%-Achse stauchte die Fehlerquoten-Serie auf ~1/3 der
Chart-Höhe, 100%-Accept-Punkte aus n=1-3-Läufen waren optisch nicht von
belastbaren unterscheidbar, der Punkt-Tooltip zeigte nur die Fehlerquote,
die Legende nur die Zweitserie. Fix (4 Teile): (a) Overlay zuschaltbar,
Default AUS (`_acceptOverlayOn`, localStorage) -- aus bleibt die Achse
hall-adaptiv wie vor dem PR; (b) Punkt-Tooltip zeigt beide Werte; (c)
Legende benennt beide Serien + unterschiedliche Basis; (d) Accept-Punkte
mit accept_n<20 gedimmt.

Dieser Test führt die ECHTE _trendChart-Funktion (+ Abhängigkeiten _fmtDE/
_autoUnit) aus dem Dashboard-HTML in Node aus (Extraktion per Klammer-
Balancer, kein Nachbau/keine Kopie der Logik).

`internal/dashboard/eval_dashboard.html` enthält ein bewusstes NUL-Byte —
Zugriff ausschließlich über `Path.read_text(encoding="utf-8")`, nie über
bash grep/sed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_HTML_PATH = Path(__file__).resolve().parents[2] / "internal" / "dashboard" / "eval_dashboard.html"


def _extract_js_function(text: str, name: str) -> str:
    start = text.index(f"function {name}(")
    brace_start = text.index("{", start)
    depth = 0
    i = brace_start
    while True:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[start : i + 1]


def _run_trend_chart(hall, versions, unit="%", accept=None, label2="Akzeptanzquote", accept_n=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node nicht verfügbar")

    html = _HTML_PATH.read_text(encoding="utf-8")
    fn_names = ["_fmtDE", "_autoUnit", "_trendChart", "_escHTML"]
    fns = "\n".join(_extract_js_function(html, n) for n in fn_names)

    script = f"""
    function cssv(v) {{ return '#000'; }}
    {fns}
    const svg = _trendChart({json.dumps(hall)}, {json.dumps(versions)}, '#c0392b', {json.dumps(unit)}, {json.dumps(accept)}, {json.dumps(label2)}, {json.dumps(accept_n)});
    process.stdout.write(svg);
    """
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"node stderr: {result.stderr}"
    return result.stdout


def test_overlay_renders_second_dashed_series_when_accept_provided():
    svg = _run_trend_chart(
        hall=[5.0, 8.0, 12.0],
        versions=["v1", "v2", "v3"],
        accept=[90.0, 70.0, 40.0],
    )
    assert "stroke-dasharray" in svg
    assert "Akzeptanzquote" in svg


def test_overlay_absent_by_default_no_second_series():
    """Regressions-Wächter: OHNE points2 (alle anderen KPI-Kacheln -- cov,
    accept selbst, dur, tokens, cost) bleibt das Chart exakt wie zuvor,
    kein zweiter Pfad/keine Legende."""
    svg = _run_trend_chart(hall=[5.0, 8.0, 12.0], versions=["v1", "v2", "v3"], accept=None)
    assert "stroke-dasharray" not in svg
    assert "Akzeptanzquote" not in svg


def test_overlay_absent_when_unit_is_not_percent():
    """Eine gemeinsame Y-Achse ergibt nur bei gleicher Einheit (%) Sinn --
    z. B. die Dauer-Kachel (Einheit h) darf nicht versehentlich eine
    %-Akzeptanzquote auf ihrer eigenen Skala einzeichnen."""
    svg = _run_trend_chart(hall=[1.0, 2.0, 1.5], versions=["v1", "v2", "v3"], unit="h", accept=[90.0, 70.0, 40.0])
    assert "stroke-dasharray" not in svg


def test_overlay_scale_accommodates_accept_values_above_hall_range():
    """Die gemeinsame Skala muss BEIDE Serien abdecken -- ein Akzeptanzwert
    weit ueber dem Fehlerquote-Wertebereich darf nicht ausserhalb des
    sichtbaren SVG-Bereichs (0..90 y, s. _trendChart-Konstanten) landen."""
    svg = _run_trend_chart(hall=[2.0, 3.0, 2.5], versions=["v1", "v2", "v3"], accept=[95.0, 92.0, 90.0])
    # Alle y-Koordinaten im Pfad (M/L-Kommandos) muessen im SVG-Canvas liegen
    # (padT=16 .. padT+ih=70 laut Konstanten in _trendChart).
    import re

    coords = re.findall(r"[ML]([\d.]+),([\d.]+)", svg)
    assert coords, "Kein Pfad gefunden"
    ys = [float(y) for _, y in coords]
    assert all(0 <= y <= 90 for y in ys), f"y-Koordinaten ausserhalb des Canvas: {ys}"


def test_overlay_mismatched_length_ignored_no_crash():
    """points2 mit abweichender Laenge (Server-Datenfehler oder Version-
    Mismatch) darf nicht crashen -- wird ignoriert (has2-Guard)."""
    svg = _run_trend_chart(hall=[5.0, 8.0, 12.0], versions=["v1", "v2", "v3"], accept=[90.0, 70.0])
    assert "stroke-dasharray" not in svg


# ── Nachbesserung (a): Legende benennt BEIDE Serien ────────────────────────


def test_overlay_legend_names_both_series():
    """Vorher nur die Zweitserie benannt -- unklar, wofuer die durchgezogene
    Linie steht. Legende zeigt jetzt "Fehlerquote" UND den Zweitserien-Namen."""
    svg = _run_trend_chart(hall=[5.0, 8.0, 12.0], versions=["v1", "v2", "v3"], accept=[90.0, 70.0, 40.0])
    assert "Fehlerquote" in svg
    assert "Akzeptanzquote" in svg


# ── Nachbesserung (b): Punkt-Tooltip zeigt beide Werte ─────────────────────


def test_overlay_point_tooltip_shows_both_values():
    """Der Tooltip an einem Datenpunkt zeigte bisher nur die Fehlerquote --
    bei aktivem Overlay fehlte der Akzeptanzwert derselben Version komplett.
    Beide Werte muessen jetzt im selben Tooltip-Text stehen."""
    svg = _run_trend_chart(hall=[5.0, 8.0, 12.0], versions=["v1", "v2", "v3"], accept=[90.0, 70.0, 40.0])
    assert "Akzeptanz" in svg
    # Letzter Punkt: Fehlerquote 12,0% UND Akzeptanz 40,0% im selben Label.
    assert "12,0%" in svg and "40,0%" in svg


def test_overlay_point_tooltip_unchanged_without_overlay():
    """Regressionswaechter: ohne Overlay (points2=None) bleibt der Tooltip-
    Text exakt wie vorher (nur die Fehlerquote, kein "Akzeptanz")."""
    svg = _run_trend_chart(hall=[5.0, 8.0, 12.0], versions=["v1", "v2", "v3"], accept=None)
    assert "Akzeptanz " not in svg  # "Akzeptanzquote" (Legende) waere ohnehin abwesend


# ── Nachbesserung (d): Accept-Punkte mit accept_n<20 gedimmt ───────────────


def test_overlay_dims_points_with_low_accept_n():
    """100%-Accept-Punkte aus n=1-3-Laeufen sahen bisher optisch nicht anders
    aus als belastbare Werte (n>=20) -- Punkte mit accept_n<20 werden jetzt
    gedimmt (reduzierte Opacity) dargestellt."""
    svg = _run_trend_chart(
        hall=[5.0, 8.0, 12.0],
        versions=["v1", "v2", "v3"],
        accept=[90.0, 70.0, 40.0],
        accept_n=[3, 25, 30],
    )
    assert 'opacity="0.35"' in svg  # v1 (n=3) gedimmt
    # v2/v3 (n>=20) nicht gedimmt -- mindestens 2 volle Kreise ohne Dimmen.
    assert svg.count('opacity="1"') >= 2


def test_overlay_no_dimming_when_accept_n_not_provided():
    """Rueckwaertskompatibel: ohne accept_n (aeltere Aufrufer/Tests) wird
    NICHTS gedimmt -- kein Crash, kein falsches Dimmen."""
    svg = _run_trend_chart(hall=[5.0, 8.0, 12.0], versions=["v1", "v2", "v3"], accept=[90.0, 70.0, 40.0])
    assert 'opacity="0.35"' not in svg


# ── Frontend-Anker: Aufrufer-Verdrahtung (nur Fehlerquote-Kachel) ──────────


def test_html_hall_tile_wires_accept_as_second_series():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("const overlayAvailable =")
    end = html.index("\n", html.index("_trendChart(vals, vers, sparkColor", start))
    block = html[start:end]
    assert "kd.key === 'hall'" in block
    assert "_acceptOverlayOn" in block
    assert "kpiTrend?.accept" in block
    assert "kpiTrend?.accept_n" in block
    assert "_trendChart(vals, vers, sparkColor, kd.u, accVals, 'Akzeptanzquote', acceptNVals)" in block


# ── Nachbesserung (a): Toggle zuschaltbar, Default AUS, localStorage ───────


def test_html_overlay_toggle_defaults_to_off_via_localstorage():
    """Default AUS: `_acceptOverlayOn` liest aus localStorage, Startwert nur
    bei explizitem 'true' aktiv (analog theme-mode-Konvention)."""
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    assert "let _acceptOverlayOn = localStorage.getItem('accept-overlay-on') === 'true'" in html


def test_html_overlay_toggle_function_persists_state():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function _toggleAcceptOverlay")
    end = html.index("\n}", start)
    block = html[start:end]
    assert "localStorage.setItem('accept-overlay-on'" in block
    assert "renderKpis" in block


def test_html_overlay_toggle_checkbox_only_for_hall_tile():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("const overlayToggleHtml")
    end = html.index("const overlayCaveatHtml", start)
    block = html[start:end]
    assert "overlayAvailable" in block
    assert "spark-overlay-toggle" in block
    assert 'onchange="_toggleAcceptOverlay(this.checked)"' in block


# ── Nachbesserung (c): Caveat zur unterschiedlichen Basis ──────────────────


def test_html_overlay_caveat_mentions_different_basis():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("const overlayCaveatHtml")
    end = html.index("const inner = document.getElementById('spark-inner')", start)
    block = html[start:end]
    assert "geroutete" in block
    assert "evaluierte" in block
    assert "accept_n" in block or "gerouteten Notes sind gedimmt" in block
