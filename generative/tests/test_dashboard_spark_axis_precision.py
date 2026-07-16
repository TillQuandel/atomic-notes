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


# ── #292-LOW: Eskalation fuer %- und h-Einheiten konsistent zur $-Eskalation ─
#
# Befund: die Eskalations-Schleife in _trendChart ersetzte kollidierende
# Labels IMMER durch `_fmtDE(v, extra) + (au.u || '')` -- eine generische
# Formel, die zufaellig zur Basis-Formatierung von "$"/generischen Einheiten
# passt (beide haengen den Suffix direkt an), aber NICHT zur Basis-
# Formatierung von "%" (haengt in `_autoUnit` gar keinen Suffix an -- Labels
# ohne Eskalation zeigen KEIN "%") und "h" (< 1h zeigt "N min", die
# Eskalation ignorierte das und zeigte immer rohe Stunden-Dezimalzahlen +
# "h" -- Sprung von z. B. "2 min" auf "0,03h" bei Kollision). Fix:
# `_autoUnit`s fmt-Funktionen nehmen jetzt einen optionalen Nachkommastellen-
# Parameter; die Eskalation ruft `au.fmt(v, extra)` statt einer zweiten,
# unit-fremden Formel -- SSoT, automatisch konsistent fuer jede Einheit.


def test_percent_sparkline_base_label_has_percent_suffix():
    """Vor dem Fix: die Basis-%-Formatierung (au.fmt ohne Eskalation) liess
    das %-Zeichen komplett weg ("3,0" statt "3,0%") -- inkonsistent zur
    Eskalation, die (zufaellig) IMMER einen Suffix anhaengt."""
    labels = _hairline_labels_via_node([3.0, 10.0, 17.0], "%")
    assert all(label.endswith("%") for label in labels), f"%-Suffix fehlt: {labels}"


def test_percent_sparkline_tight_span_escalates_to_distinct_labels():
    labels = _hairline_labels_via_node([5.001, 5.004, 5.007], "%")
    assert len(set(labels)) == 3, f"Achsen-Labels nicht unterscheidbar: {labels}"
    assert all(label.endswith("%") for label in labels)


def test_hour_sparkline_sub_1h_tight_span_stays_in_minutes_convention():
    """Vor dem Fix: sub-1h-Werte mit kollidierenden gerundeten Minuten
    eskalierten auf rohe Stunden-Dezimalzahlen ("0,03h") statt einfach mehr
    Nachkommastellen in derselben (Minuten-)Konvention zu zeigen -- inkon-
    sistent zur Basis-Anzeige direkt daneben ("2 min")."""
    labels = _hairline_labels_via_node([0.030, 0.034, 0.038], "h")
    assert len(set(labels)) == 3, f"Achsen-Labels nicht unterscheidbar: {labels}"
    assert all(label.endswith("min") for label in labels), f"Einheiten-Sprung zu Stunden: {labels}"


def test_hour_sparkline_over_1h_tight_span_escalates_in_hours():
    # 2.0-Bereich statt 1.0: bei knapp über 1h zieht das 20%-Padding
    # (_trendChart: pad = (vmax-vmin)*0.2) die untere Hilfslinie sonst unter
    # die 1h-Grenze -- dann korrekt gemischte Einheiten (s. Docstring unten),
    # kein Bug. Hier bleiben alle 3 Hilfslinien auch MIT Padding über 1h.
    labels = _hairline_labels_via_node([2.001, 2.004, 2.007], "h")
    assert len(set(labels)) == 3, f"Achsen-Labels nicht unterscheidbar: {labels}"
    assert all(label.endswith("h") for label in labels)


def test_hour_sparkline_straddling_1h_boundary_mixes_units_correctly():
    """Kein Bug, sondern Design-Konsequenz: liegt die untere Hilfslinie NACH
    Padding-Abzug unter 1h, zeigt sie ("min") einheitenkorrekt an, waehrend
    die oberen weiter in Stunden eskalieren -- dieselbe Logik wie die Basis-
    Anzeige (_autoUnit: `v < 1 ? min : h`), nicht vereinheitlicht auf eine
    Einheit. Regressions-Wächter fuer genau diesen Grenzfall."""
    labels = _hairline_labels_via_node([1.001, 1.004, 1.007], "h")
    assert len(set(labels)) == 3, f"Achsen-Labels nicht unterscheidbar: {labels}"
    assert labels[0].endswith("min")
    assert labels[1].endswith("h") and labels[2].endswith("h")


def test_hour_sparkline_base_label_unchanged_rounded_minutes():
    """Regressions-Wächter: die BASIS-Anzeige (kein Eskalationsbedarf) bleibt
    exakt wie zuvor -- ganzzahlige Minuten, keine Nachkommastelle. Bekannter
    Referenzwert (unveraendert vor/nach Fix, per Node-Lauf verifiziert):
    hairVals[0] (untere Hilfslinie) rundet bei diesem Wertebereich auf 0."""
    labels = _hairline_labels_via_node([0.05, 0.5, 0.9], "h")
    assert labels[0] == "0 min"
