"""Tests für #237: 0-Eval-/Merge-only-Lauf liest sich wie Erfolgslauf.

Befund: `avg_accept` (eval_dashboard.py `_calc_kpis`) wird aus Vault-/
Generated-Zaehlern berechnet — unabhaengig davon, ob ueberhaupt Notes
evaluiert wurden (`n_notes`). Bei einem Merge-only-Lauf ohne Eval-Stage
(z. B. Run 20260713-084724: `avg_accept=100.0, n_notes=0`) zeigte die
Akzeptanz-Kachel dadurch ein gruenes "100,0 %", waehrend das Empty-Banner
gleichzeitig unbedingt behauptete "Kennzahlen und Charts sind leer" —
sachlich falsch, weil die Betriebskennzahlen (Strip, Akzeptanz-Basis)
nicht leer waren, nur die Qualitaets-Charts (Eval-basiert: Fehlerquote/
Belegrate). Fix: Kachel bei n_notes==0 neutralisieren (grau/– statt
kpi-good-gruen), Banner-Text ehrlich konditionalisieren.
"""

from __future__ import annotations

from generative.eval_dashboard_server import _build_live_html


def test_accept_tile_neutralized_when_no_notes_evaluated():
    html = _build_live_html()
    # Guard muss VOR der bisherigen tone-Berechnung greifen (n_notes==0 -> neutral),
    # unabhaengig vom Zahlenwert von avg_accept.
    assert "kpis.n_notes===0" in html


def test_empty_banner_text_has_stable_id_for_dynamic_update():
    html = _build_live_html()
    assert 'id="empty-banner-text"' in html


def test_empty_banner_conditionalizes_on_operational_data():
    html = _build_live_html()
    # Ehrlicher Text, wenn Betriebskennzahlen (Akzeptanz-Basis) trotz 0 Evals
    # vorhanden sind — "Charts sind leer" darf dann nicht mehr behauptet werden.
    assert "nur Betriebskennzahlen" in html
    assert "kpi_accept_n" in html


def test_empty_banner_notes_matrix_stays_filled(monkeypatch):
    """#322: die Versions×PDF-Matrix ignoriert JEDEN aktiven Filter (bewusster
    Design-Entscheid, test_pair_matrix_ignores_active_single_value_pdf_and_
    version_filters) und bleibt bei 0-Notes-Filterkombinationen gefuellt --
    das Banner darf "Charts sind leer" darum nicht mehr unbedingt behaupten,
    weder im statischen Default-Text noch in den beiden dynamischen JS-
    Varianten."""
    html = _build_live_html()
    matrix_hint = "außer der Versions×PDF-Matrix"
    # (a) statischer Default-Text (vor dem ersten Datenladen)
    default_start = html.index('id="empty-banner-text"')
    default_end = html.index("</span>", default_start)
    assert matrix_hint in html[default_start:default_end]
    # (b) beide dynamischen JS-Textvarianten (hasOperationalData true/false)
    js_start = html.index("const hasOperationalData")
    js_end = html.index("\n    }", js_start)
    js_block = html[js_start:js_end]
    assert js_block.count(matrix_hint) == 2
