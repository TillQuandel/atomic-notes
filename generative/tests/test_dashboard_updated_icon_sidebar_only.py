"""Tests für Till-Entscheid 2026-07-16: Zeitstempel nur noch in der Sidebar.

Ist-Zustand vor diesem Fix: Oben rechts in der Filterbar-Zeile zeigte ein
↻-Hover-Icon (`id="updated-at"`) den `generated_at`-Zeitstempel per Tooltip
(Feature #223). Gleichzeitig zeigte die Sidebar im Fuß (`.side-foot`) bereits
permanent Datum/Uhrzeit (`#sf-time`), die aktive eval_version (`#sf-ev-
select`) und `auto-refresh 15 s` — Dopplung derselben Information an zwei
Stellen.

Fix: Das Kopfzeilen-Icon entfällt ersatzlos (SSoT ist die Sidebar). Bei
eingeklappter Sidebar sind die vollen Meta-Zeilen im Fuß aber versteckt
(`body.collapsed .side-foot > *:not(.theme-toggle)`) — als Ersatz zeigt der
Fuß dann ein kompaktes ↻-Symbol (`.sf-collapsed-hint`) mit Hover/Focus-
Tooltip (gleiche hint/hint-pop-Mechanik wie überall im Dashboard), das
Datenstand + Refresh-Hinweis + aktive eval_version kompakt zusammenfasst.

`internal/dashboard/eval_dashboard.html` enthält ein bewusstes NUL-Byte —
Zugriff ausschließlich über `_build_live_html()`/`Path.read_text(encoding=
"utf-8")`, nie über bash grep/sed.
"""

from __future__ import annotations


def test_html_header_updated_icon_removed():
    """Das ↻-Hover-Icon oben rechts in der Filterbar-Zeile ist komplett weg —
    weder das Element selbst noch die id, über die JS es befüllt hat."""
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    assert 'id="updated-at"' not in html
    assert 'class="updated hint"' not in html
    # Engerer Kopfzeilen-Anker (_build_live_html-Muster wie im
    # eval_version-Dropdown-Test): kein Hint-Trigger-Element (nur noch ein
    # erklaerender Kommentar, der "↻" beschreibend erwaehnt) im Filterbar-
    # Abschnitt selbst.
    section = html[html.index('<div class="filterbar">') : html.index('id="filter-badges"')]
    assert '<span class="updated' not in section
    assert "hint-pop" not in section


def test_html_sidebar_foot_has_collapsed_hint():
    """Ersatz für das entfernte Kopfzeilen-Icon: kompaktes ↻-Symbol im
    Sidebar-Fuß (nur sichtbar wenn body.collapsed, s. CSS), mit eigenem
    hint-pop-Tooltip. #sf-time (die volle, expanded Zeitzeile) bleibt
    unverändert bestehen — SSoT, nicht Teil dieses Fixes."""
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    side_foot = html[html.index('class="side-foot"') : html.index("</aside>")]
    assert 'class="sf-collapsed-hint hint"' in side_foot
    assert 'id="sf-collapsed-info"' in side_foot
    assert 'id="sf-time"' in side_foot


def test_html_render_with_data_wires_collapsed_hint_and_keeps_sf_time():
    """JS-Verdrahtung: sf-time (expanded) bleibt unverändert befüllt; neu ist
    sf-collapsed-info (collapsed Fuß-Tooltip) mit Datenstand + Refresh-Hinweis
    + aktiver eval_version. Das frühere Kopfzeilen-Icon (id=updated-at) taucht
    im Render-Code nicht mehr auf."""
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function _renderWithData")
    end = html.index("Loading-Skeletons entfernen", start)
    block = html[start:end]
    assert "getElementById('sf-time')" in block
    assert "getElementById('sf-collapsed-info')" in block
    assert "d.eval_version" in block
    assert "updated-at" not in block
