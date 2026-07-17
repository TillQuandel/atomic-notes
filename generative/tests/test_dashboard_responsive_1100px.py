"""Punkt 9 (U8/U9, Multi-Perspektiven-Review 2026-07-15): Responsive-Fixes.

U8: bei <=1200px (kpis-perf: 4 Spalten) UND <=900px (2 Spalten) liess die
5. (letzte) KPI-Kachel eine Zeile allein mit leeren Spalten daneben stehen
(5%4=1 bzw. 5%2=1) -- sichtbare Lücke statt sauberem Umbruch. Fix: letzte
Kachel spannt die volle Breite, wenn sie allein in ihrer Zeile steht.

U9: das Modell-Filter brach als letzte Filter-Group allein in eine neue
Zeile -- kompaktere Gruppierung durch engeres .filter-group-Padding und
schmalere .fselect-Maximalbreite, urspruenglich bei <=1200px.

#324 (Folge-Fund UX-Review 16.07.): der naechste Breakpoint war @media
(min-width: 1201px) (Matrix-Sticky-Regel, andere Zustaendigkeit) -- dazwischen
(1201–~1430px) blieb die volle unkompaktierte Breite aktiv, "Modell" brach bei
1280px allein in Zeile 2 um. Fix: die U9-Kompakt-Styles in eine eigene, auf
<=1440px erweiterte Media-Query verschoben (die kpis-perf-Regeln bleiben
bewusst bei <=1200px, davon nicht betroffen).

Playwright-Sichtprüfung (isolierter Testserver, Live-Daten read-only,
1100px): kpis-perf letzte Kachel spannt jetzt die volle Breite (kein
206px-Rest-Tile mit Lücke mehr), Filterbar-Gruppen kompakter. Diese Datei
sichert die CSS-Regeln als Regressionswächter (HTML-Anker-Muster, kein
bash grep/sed auf dem bewussten NUL-Byte).
"""

from __future__ import annotations

from generative.eval_dashboard_server import _build_live_html


def _responsive_css_block(max_width: int) -> str:
    html = _build_live_html()
    start = html.index(f"@media (max-width: {max_width}px)")
    end = html.index("\n}", start)
    return html[start:end]


def test_kpis_perf_last_tile_spans_full_row_at_1200px_breakpoint():
    block = _responsive_css_block(1200)
    assert ".kpis-perf .kpi:last-child" in block
    assert "grid-column: 1 / -1" in block


def test_kpis_perf_last_tile_spans_full_row_at_900px_breakpoint():
    block = _responsive_css_block(900)
    assert ".kpis-perf .kpi:last-child" in block
    assert "grid-column: 1 / -1" in block


def test_filter_group_padding_compacted_at_1440px_breakpoint():
    """#324: auf <=1440px erweitert, schliesst die 1201–1430px-Luecke."""
    block = _responsive_css_block(1440)
    assert ".filter-group { padding: 0 10px; }" in block


def test_fselect_max_width_reduced_at_1440px_breakpoint():
    """#324: auf <=1440px erweitert, schliesst die 1201–1430px-Luecke."""
    block = _responsive_css_block(1440)
    assert ".fselect { max-width: 150px; }" in block


def test_filter_group_compaction_no_longer_gated_at_1200px():
    """Regressions-Waechter (#324): die Kompakt-Styles duerfen NICHT mehr im
    engeren 1200px-Block stehen -- sonst waere die 1201–1430px-Luecke wieder
    da, nur unbemerkt durch eine zufaellige Duplizierung."""
    block = _responsive_css_block(1200)
    assert ".filter-group { padding: 0 10px; }" not in block
    assert ".fselect { max-width: 150px; }" not in block
