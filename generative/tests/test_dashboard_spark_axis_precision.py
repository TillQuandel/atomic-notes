"""Multi-Perspektiven-Review 2026-07-15, Bug 3 (UX-Fund U2):

Achsenbeschriftung der Spark-Trend-Charts (`_trendChart`,
internal/dashboard/eval_dashboard.html) rundete den Rohwert ERST über
`_fmtDE()` (DE-Locale-String MIT Dezimalkomma, z. B. "0,03") und parste das
Ergebnis danach per `parseFloat()` zurück. `parseFloat("0,03")` bricht am
Komma ab und liefert `0` (kein Komma-Dezimaltrennzeichen in JS) — alle 3
Hilfslinien-Labels kollabierten auf denselben gerundeten Wert (Befund: API-
Kosten-Sparkline zeigte 3x "0,00$", obwohl die Tooltips daneben korrekt
0,03–0,15 $ zeigten, weil deren Formatierung nicht über diesen Roundtrip lief).

Dieser Test führt die ECHTEN JS-Funktionen (_fmtDE, _autoUnit, _trendChart)
aus dem Dashboard-HTML in Node aus (Extraktion per Klammer-Balancer, kein
Nachbau/keine Kopie der Logik) — echter Verhaltensbeleg statt reiner
Text-Grep, weil der Bug numerisch ist (Rundung/Locale), nicht rein textuell.

`internal/dashboard/eval_dashboard.html` enthält ein bewusstes NUL-Byte
(Drawer-Key-Separator) — Zugriff ausschließlich über
`Path.read_text(encoding="utf-8")`, nie über bash grep/sed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_HTML_PATH = Path(__file__).resolve().parents[2] / "internal" / "dashboard" / "eval_dashboard.html"


def _extract_js_function(text: str, name: str) -> str:
    """Extrahiert `function <name>(...) { ... }` per Klammer-Balancer (keine
    Kopie/Umschreibung der Logik — derselbe Quelltext wird in Node evaluiert)."""
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


def _hairline_labels_via_node(points: list[float], unit: str) -> list[str]:
    """Ruft die echte `_trendChart`-Funktion aus dem HTML mit Test-Daten auf
    und gibt die 3 Hilfslinien-Achsenlabels zurück (Reihenfolge min/mid/max)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node nicht verfügbar")

    html = _HTML_PATH.read_text(encoding="utf-8")
    fmt_de = _extract_js_function(html, "_fmtDE")
    auto_unit = _extract_js_function(html, "_autoUnit")
    trend_chart = _extract_js_function(html, "_trendChart")

    script = f"""
    function cssv(v) {{ return '#000'; }}
    {fmt_de}
    {auto_unit}
    {trend_chart}
    const svg = _trendChart({json.dumps(points)}, ["v1", "v2"], "#000", {json.dumps(unit)});
    const labels = [...svg.matchAll(/opacity=".7">([^<]+)<\\/text>/g)].map(m => m[1]);
    process.stdout.write(JSON.stringify(labels));
    """
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"node stderr: {result.stderr}"
    return json.loads(result.stdout)


def test_cost_sparkline_hairline_labels_are_not_all_identical():
    # Realer Befund-Wertebereich (API-Kosten-Kachel, #Bug3-Beleg): 0,03–0,15 $.
    labels = _hairline_labels_via_node([0.03, 0.09, 0.15, 0.06, 0.15], "$")
    assert len(labels) == 3
    assert len(set(labels)) == 3, f"Achsen-Labels nicht unterscheidbar: {labels}"


def test_cost_sparkline_hairline_labels_are_not_all_zero():
    labels = _hairline_labels_via_node([0.03, 0.09, 0.15, 0.06, 0.15], "$")
    assert labels != ["0,00$", "0,00$", "0,00$"]


def test_tight_span_still_yields_distinct_labels():
    """Sehr enge Spanne (< 0,01 Differenz) — Nachkommastellen muessen bei
    Bedarf erhoehen, sonst kollabieren die Labels erneut auf denselben Wert."""
    labels = _hairline_labels_via_node([0.0301, 0.0304, 0.0307], "$")
    assert len(set(labels)) == 3, f"Achsen-Labels nicht unterscheidbar: {labels}"
