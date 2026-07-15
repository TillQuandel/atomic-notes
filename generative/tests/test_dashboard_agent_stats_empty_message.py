"""Multi-Perspektiven-Review 2026-07-15, Bug 2 (UX-Fund U1):

Agent-Statistiken (ch7/ch8) kommen ausschließlich aus lokalen Trace-JSONLs
(.cache/runs/<run_id>.jsonl) — es gibt keine Agent-Ebene in der SQLite-DB
(pipeline_runs/note_evals kennen nur Run-Summen, kein "je Agent"). Ohne
JSONL-Traces liefert `_read_agent_stats()` `{}` und das Frontend zeigte die
generische Filter-Leer-Meldung ("Keine Daten für diese Filter-Kombination"),
obwohl gar kein Filter aktiv war — irreführend, weil es keinen Filter zum
Ändern gibt, der das beheben würde.

Fix: Frontend unterscheidet — kein Filter aktiv + agent_stats leer → ehrliche
Meldung, dass lokale Lauf-Traces fehlen. Filter aktiv + agent_stats leer →
bisherige (weiterhin korrekte) Filter-Meldung.

`internal/dashboard/eval_dashboard.html` enthält ein bewusstes NUL-Byte
(Drawer-Key-Separator) — Zugriff ausschließlich über
`_build_live_html()`/`Path.read_text(encoding="utf-8")`, nie über
bash grep/sed.
"""

from __future__ import annotations

from generative.eval_dashboard_server import _build_live_html


def _ch7_ch8_block(html: str) -> str:
    start = html.index("// ch7 + ch8 — Agents")
    end = html.index("/* ── Scatter", start)
    return html[start:end]


def test_honest_no_traces_message_defined():
    html = _build_live_html()
    assert "Agent-Statistiken benötigen lokale Lauf-Traces" in html
    assert ".cache/runs" in html


def test_ch7_ch8_block_distinguishes_filter_active_from_no_traces():
    html = _build_live_html()
    block = _ch7_ch8_block(html)
    # Muss dieselbe globale Filter-Erkennung nutzen wie der Top-Level-Banner
    # (Zeile ~2358) — sonst rät die Meldung nur, statt den echten Zustand zu prüfen.
    assert "_globalFilters" in block
    assert "Agent-Statistiken benötigen lokale Lauf-Traces" in block


def test_filtered_empty_state_still_falls_back_to_generic_message():
    """Bei aktivem Filter bleibt die alte, weiterhin zutreffende Meldung
    (echte 0-Treffer-Filterkombination) — setChartEmpty() liefert sie per
    Default (undefined msg), keine Regression durch den Fix."""
    html = _build_live_html()
    block = _ch7_ch8_block(html)
    assert "setChartEmpty('ch7'" in block
    assert "setChartEmpty('ch8'" in block
