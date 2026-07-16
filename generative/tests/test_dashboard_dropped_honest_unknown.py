"""Punkt 6 (D5, Reviews 15.07.): Dropped-Zaehlung existiert nur im DB-Fallback-Pfad.

Befund: `_calc_kpis`s `total_dropped` summiert `r.get("n_dropped", 0)` ueber
`all_log_runs`. Die PRIMAERE Datenquelle (`_read_all_log_runs()`, Log-Dateien)
baut ihre Zeilen aus `_NOTE_RE` ("[DRY-RUN] -> (Vault|Inbox)...")-Treffern --
verworfene Kandidaten (nie bis zum Draft gekommen) hinterlassen dort gar
keine Zeile und tauchen im `notes`-Dict nie auf. `n_dropped` ist an dieser
Quelle STRUKTURELL nicht ermittelbar, nicht nur zufaellig leer. Trotzdem
liefert `.get("n_dropped", 0)` in diesem Fall still `0` -- ununterscheidbar
von "wirklich 0 verworfen". Nur der DB-Fallback-Pfad
(`eval_dashboard_server.py`, `pipeline_runs`-Tabelle) setzt `n_dropped`
tatsaechlich (auch als echte 0, wenn die Spalte 0 ist).

Fix: `_calc_kpis` liefert `total_dropped=None` (nicht 0), wenn KEINE Zeile in
`all_log_runs` den Key `n_dropped` traegt (Primaer-/Log-Pfad) -- der Client
zeigt dafuer ehrlich "–" mit Tooltip statt der stillen 0. Traegt mindestens
eine Zeile den Key (DB-Fallback-Pfad), bleibt die bisherige Summe (echte 0
eingeschlossen) unveraendert."""

from __future__ import annotations

from generative.eval_dashboard import _calc_kpis


def _log_run(**over):
    """Primaer-Pfad-Zeile (_read_all_log_runs()) -- traegt NIE 'n_dropped'."""
    base = {
        "key": "a",
        "label": "A",
        "ver": "v1",
        "n_total": 10,
        "n_vault": 5,
        "n_merge": 2,
        "n_inbox": 3,
        "accept_pct": 50.0,
        "words": 1000,
        "pages": 5,
        "chunks": 3,
    }
    base.update(over)
    return base


def _db_fallback_run(n_dropped, **over):
    """DB-Fallback-Pfad-Zeile (eval_dashboard_server.py) -- traegt IMMER 'n_dropped'."""
    base = {
        "key": "a",
        "label": "A",
        "ver": "v1",
        "n_total": 10,
        "n_vault": 5,
        "n_merge": 0,
        "n_inbox": 5,
        "n_dropped": n_dropped,
        "n_words": 1000,
        "words": 1000,
        "pages": 0,
        "accept_pct": 50.0,
    }
    base.update(over)
    return base


def test_total_dropped_is_none_when_no_row_carries_the_key():
    """Primaer-/Log-Pfad: n_dropped strukturell nicht ermittelbar -> None,
    nicht stille 0."""
    runs = [_log_run(), _log_run(ver="v2")]
    kpis = _calc_kpis({}, runs, [], [])
    assert kpis["total_dropped"] is None


def test_total_dropped_sums_real_zero_when_db_fallback_reports_zero():
    """DB-Fallback-Pfad mit echter 0 (Spalte gesetzt, aber 0 verworfen) --
    bleibt 0, nicht None (der Unterschied IST bekannt)."""
    runs = [_db_fallback_run(0), _db_fallback_run(0)]
    kpis = _calc_kpis({}, runs, [], [])
    assert kpis["total_dropped"] == 0


def test_total_dropped_sums_nonzero_db_fallback_values():
    runs = [_db_fallback_run(3), _db_fallback_run(7)]
    kpis = _calc_kpis({}, runs, [], [])
    assert kpis["total_dropped"] == 10


def test_total_dropped_none_on_empty_all_log_runs():
    """Keine Runs ueberhaupt -- ebenfalls unbekannt (kein Pfad hat je etwas
    gemeldet), nicht implizit 0."""
    kpis = _calc_kpis({}, [], [], [])
    assert kpis["total_dropped"] is None


# ── Frontend-Anker: "–" mit Tooltip statt stiller 0 ────────────────────────


def test_html_dropped_cell_shows_dash_when_total_dropped_is_null():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    i = html.index("cell('Dropped'")
    line = html[i : html.index("\n", i)]
    assert "kpis.total_dropped != null" in line or "kpis.total_dropped !== null" in line
    assert "'–'" in line
