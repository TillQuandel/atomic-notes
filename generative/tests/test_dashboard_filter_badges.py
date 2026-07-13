"""Tests für #236: Filter-Badges + Reset strukturell unsichtbar.

Befund: `_renderFilterBadges()` setzte bei aktiven Filtern
`container.style.display = active.length ? '' : 'none';` — der leere String
`''` löscht nur den Inline-Stil und fällt auf die CSS-Basis
`.filter-badges { display: none; }` zurück. Badges + die entfernbaren
X-Chips waren dadurch strukturell unsichtbar, unabhängig davon ob Filter
aktiv sind. Zusätzlich fehlte ein außerhalb des Empty-State erreichbarer
„Alle Filter zurücksetzen"-Button.
"""

from __future__ import annotations

from generative.eval_dashboard_server import _build_live_html


def test_filter_badges_display_flex_when_active():
    html = _build_live_html()
    assert "active.length ? 'flex' : 'none'" in html
    # Regressions-Wächter: der leere-String-Fallback darf nicht wiederkehren.
    assert "active.length ? '' : 'none'" not in html


def test_filter_badges_have_always_visible_reset_button():
    # Reset muss auch außerhalb des Empty-State (0 Treffer) erreichbar sein,
    # sobald mindestens ein Filter aktiv ist.
    html = _build_live_html()
    assert "fbadge-reset" in html
    assert "resetAllFilters()" in html
