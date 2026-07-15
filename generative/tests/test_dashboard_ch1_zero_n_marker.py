"""Test für #U5 (Multi-Perspektiven-Review 2026-07-15): n=0-Balken im
Akzeptanz-Chart (ch1) — das aufgedruckte Wert-Label ("100,0 %") wirkte wie ein
echter Wert, obwohl 0 Notes evaluiert wurden (nur Routing-Daten, siehe #249).
Die Balkenfarbe war für diesen Fall bereits grau (barColor), das Wert-Label
selbst trug aber kein eigenes Signal.

Fix: bei nMap[label]==0 (dieselbe Bedingung wie barColor) wird das Wert-Label
zusätzlich per globalAlpha gedimmt UND mit einem Sternchen ("100,0 %*")
gekennzeichnet.
"""

from __future__ import annotations

from generative.eval_dashboard_server import _build_live_html


def test_ch1_value_label_marks_zero_n_bars_with_asterisk_and_dimming():
    html = _build_live_html()
    assert "isEmpty?.55:1" in html or "isEmpty ? .55 : 1" in html
    assert "isEmpty?'*':''" in html or "isEmpty ? '*' : ''" in html
    # Dieselbe n===0-Bedingung wie barColor -- kein zweiter Datenpfad.
    assert "const isEmpty = (nMap[rawPairs[i]?.l]||0)===0;" in html
