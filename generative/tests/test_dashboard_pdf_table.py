"""Tests für die per-PDF-Tabelle des Eval-Dashboards (#194).

Die Tabelle stand strukturell auf drei Datengrundlagen in einer Zeile:
version/accept aus den Routing-Logs (log_data), hall/cov über ALLE Versionen
gepoolt (Substring-Match auf quality_rows), n_notes = Eval-Instanzen. Das ergab
Bates-Dreifachzeilen (pdf_key-Drift), fehlende PDFs (nur in quality_rows) und
version-gemischte Kennzahlen.

Fixtures reproduzieren die vier Hauptbefunde synthetisch (RED vor dem Fix):
  (a) Zeile mischt Versionen        -> version + hall aus EINER Version
  (b) Substring-Doppelzählung Bates -> genau EINE kanonische Zeile
  (c) PDF nur in quality_rows       -> erscheint (Union-Iteration)
  (d) n_notes zählt Re-Evals        -> distinct Notes

Alle Fixtures sind rein synthetisch; keine .cache-Daten werden gelesen.
"""

from __future__ import annotations

from generative.eval_dashboard import (
    _build_log_data,
    _calc_pdf_table,
    _chart_acceptance,
    _distinct_notes,
    _pdf_group_key,
    _render_pdf_table,
)

_CUR = "v0.3.140"  # deterministischer Config-Anker für den Orphan-Guard (#191)


def _q(pdf, note, ver, hall, cov):
    """Minimale note_evals-Zeile."""
    return {
        "pdf": pdf,
        "note_path": note,
        "pipeline_version": ver,
        "hallucination_rate": hall,
        "coverage_factual": cov,
    }


def _run(key, label, ver, n_total, n_vault, words=None):
    """Minimaler Routing-Run (pipeline_runs/log)."""
    return {
        "key": key,
        "label": label,
        "ver": ver,
        "n_total": n_total,
        "n_vault": n_vault,
        "accept_pct": round(n_vault / n_total * 100, 1) if n_total else 0.0,
        "words": words,
    }


# ── (b) Substring-Doppelzählung + pdf_key-Drift ─────────────────────────────
def _bates_fixture():
    quality_rows = [
        # Volltitel-Namensraum (note_evals)
        _q("Bates - 2017 - Information Behavior.pdf", "b_n1", "v0.3.10", 0.05, 0.80),
        _q("Bates - 2017 - Information Behavior.pdf", "b_n1", "v0.3.20", 0.10, 0.70),  # Re-Eval derselben Note
        # Kebab-Namensraum derselben Quelle (neueste Version)
        _q("bates-2017.pdf", "b_n2", "v0.3.20", 0.20, 0.60),
    ]
    all_log_runs = [
        _run("bates", "Bates", "v0.3.10", 5, 5, 1000),
        _run("bates-2017", "bates-2017", "v0.3.20", 4, 2, 1000),
        # Toter Triple-Dash-Log-Key ohne Eval-Daten (0 % Routing, 0 Notes)
        _run("bates---2017---information-behavior", "bates---2017---information-behavior", "v0.3.05", 1, 0),
    ]
    return quality_rows, all_log_runs


def test_bates_variants_collapse_to_single_row():
    """(b) Drei pdf_key-Varianten derselben Quelle -> genau eine Tabellenzeile."""
    quality_rows, all_log_runs = _bates_fixture()
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version=_CUR)
    bates_rows = [r for r in rows if "bates" in _pdf_group_key(r["key"])]
    assert len(bates_rows) == 1, f"Erwartet 1 Bates-Zeile, bekam {len(bates_rows)}: {[r['key'] for r in bates_rows]}"


def test_row_version_and_hall_from_single_newest_version():
    """(a) version + hall stammen aus GENAU der neuesten Eval-Version, nicht gepoolt."""
    quality_rows, all_log_runs = _bates_fixture()
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version=_CUR)
    row = next(r for r in rows if "bates" in _pdf_group_key(r["key"]))
    assert row["version"] == "v0.3.20"
    # Nur v0.3.20-Zeilen: b_n1=0.10, b_n2=0.20 -> Mittel 15,0 %.
    # Gepoolt über alle Versionen (Bug) wäre (0.05+0.10+0.20)/3 = 11,7 %.
    assert row["hall"] == 15.0, f"hall={row['hall']} (erwartet 15.0 = nur neueste Version)"


def test_n_notes_counts_distinct_notes_not_re_evals():
    """(d) n_notes = distinct Notes der neuesten Version, nicht Eval-Instanzen."""
    quality_rows = [
        _q("Porst-2014-Auszug.pdf", "p1", "v0.3.20", 0.10, 0.7),
        _q("Porst-2014-Auszug.pdf", "p1", "v0.3.20", 0.12, 0.7),  # Re-Eval derselben Note
        _q("Porst-2014-Auszug.pdf", "p1", "v0.3.20", 0.08, 0.7),  # Re-Eval derselben Note
        _q("Porst-2014-Auszug.pdf", "p2", "v0.3.20", 0.15, 0.7),
    ]
    all_log_runs = [_run("porst-2014-auszug", "Porst", "v0.3.20", 4, 4)]
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version=_CUR)
    row = next(r for r in rows if r["key"].startswith("porst"))
    assert row["n_notes"] == 2, f"n_notes={row['n_notes']} (4 Eval-Instanzen, 2 distinct Notes)"


def test_pdf_only_in_quality_rows_is_visible():
    """(c) PDF mit Eval-Daten aber ohne Routing-Run erscheint (Union-Iteration)."""
    quality_rows = [
        _q("Hertzum - 2023 - Information seeking.pdf", "h1", "v0.3.15", 0.04, 0.90),
        _q("Hertzum - 2023 - Information seeking.pdf", "h2", "v0.3.15", 0.05, 0.92),
    ]
    all_log_runs = []  # keine Routing-Runs für Hertzum
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version=_CUR)
    hertzum = [r for r in rows if r["key"].startswith("hertzum")]
    assert len(hertzum) == 1, f"Hertzum fehlt in der Tabelle: {[r['key'] for r in rows]}"
    assert hertzum[0]["n_notes"] == 2
    assert hertzum[0]["accept"] is None  # kein Routing-Run -> keine Akzeptanz
    assert hertzum[0]["cov"] == 91.0  # Median(90, 92)


def test_orphan_version_does_not_capture_row_version():
    """#191-Bugklasse: verwaiste WIP-Version (> Config) darf 'neueste' nicht kapern."""
    quality_rows = [
        _q("Assfalg - 2013 - Metadaten.pdf", "a1", "v0.3.140", 0.09, 0.7),
        _q("Assfalg - 2013 - Metadaten.pdf", "a2", "v0.9.99", 0.50, 0.3),  # Orphan-WIP
    ]
    all_log_runs = [_run("assfalg", "Assfalg", "v0.3.140", 2, 1)]
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version=_CUR)
    row = next(r for r in rows if r["key"].startswith("assfalg"))
    assert row["version"] == "v0.3.140", f"Orphan v0.9.99 hat die Zeile gekapert: {row['version']}"


def test_accept_column_reports_run_count():
    """Accept aus anderer Quelle (Routing) -> n der Läufe wird ausgewiesen."""
    quality_rows = [_q("Ebner - 2019 - Meta.pdf", "e1", "v0.3.140", 0.04, 0.8)]
    all_log_runs = [
        _run("ebner", "Ebner", "v0.3.140", 3, 3),
        _run("ebner", "Ebner", "v0.3.140", 4, 1),
    ]
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version=_CUR)
    row = next(r for r in rows if r["key"].startswith("ebner"))
    # Gepoolt über beide Läufe der Version: (3+1)/(3+4) = 57,1 %
    assert row["accept"] == 57.1
    assert row["accept_n"] == 2
    assert row["accept_ver"] == "v0.3.140"


# ── ins-quality-Streifen: n-Feld am Accept-Chart ────────────────────────────
def test_accept_chart_carries_n_field():
    """Der Accept-Chart-Datensatz trägt n_notes je Balken -> Streifen kann
    '0 % von n Notes' von '0 Notes' unterscheiden (Issue-Kommentar)."""
    quality_rows, all_log_runs = _bates_fixture()
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version=_CUR)
    chart = _chart_acceptance(rows)
    assert "n" in chart
    assert len(chart["n"]) == len(chart["labels"]) == len(chart["values"])
    # Bates-Balken hat echte Notes (n>0)
    assert any(n > 0 for n in chart["n"])


def test_accept_chart_dead_key_has_zero_n():
    """Ein Routing-only-PDF ohne Eval-Daten erscheint mit n=0 (Streifen filtert es)."""
    quality_rows = [_q("Real - 2020 - X.pdf", "r1", "v0.3.140", 0.05, 0.8)]
    all_log_runs = [
        _run("real", "Real", "v0.3.140", 2, 2),
        _run("deadkey-test-only", "deadkey-test-only", "v0.3.140", 1, 0),  # Routing 0 %, 0 Eval-Notes
    ]
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version=_CUR)
    chart = _chart_acceptance(rows)
    dead = [(lbl, n) for lbl, n in zip(chart["labels"], chart["n"]) if "deadkey" in _pdf_group_key(lbl)]
    assert dead, "Routing-only-Key fehlt im Accept-Chart"
    assert dead[0][1] == 0, f"Routing-only-Key sollte n=0 haben, hat {dead[0][1]}"


# ═══════════════════════════════════════════════════════════════════════════
# Nachbesserung PR #222 (#194): 7 konsolidierte Funde (Fable/Mistral/Qwen)
# ═══════════════════════════════════════════════════════════════════════════


# ── (1) Legacy-Renderpfad crasht auf fehlendem 'pages' ──────────────────────
def test_legacy_render_pdf_table_no_pages_keyerror():
    """`_render_pdf_table` (Legacy-CLI-Pfad, main→_build_html) darf nicht mit
    KeyError 'pages' crashen — `_calc_pdf_table` liefert das Feld nicht mehr."""
    quality_rows, all_log_runs = _bates_fixture()
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    html = _render_pdf_table(rows)  # RED: KeyError 'pages'
    assert "<table" in html
    assert "Seiten" not in html  # Seiten-Spalte strukturell entfernt (DB-Fallback hart 0)


# ── (2) Mehrdeutiger Kurz-Key ohne Jahr matcht mehrere Jahrgänge ────────────
def _beutelspacher_two_years():
    quality_rows = [
        _q("Beutelspacher - 2014 - Diskrete Mathematik.pdf", "bs14", "v0.3.140", 0.05, 0.80),
        _q("Beutelspacher - 2022 - Kryptografie.pdf", "bs22", "v0.3.140", 0.06, 0.70),
    ]
    # Routing-Run trägt nur den Autornamen ohne Jahr → Segment-Präfix BEIDER Gruppen
    all_log_runs = [_run("beutelspacher", "Beutelspacher", "v0.3.140", 10, 7)]
    return quality_rows, all_log_runs


def test_ambiguous_short_key_not_assigned_to_year_group():
    """(2) Ein jahrloser Autor-Key darf keiner der beiden Jahrgangs-Gruppen
    zugeschlagen werden — er läuft als eigene Routing-only-Zeile."""
    quality_rows, all_log_runs = _beutelspacher_two_years()
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    year_rows = [r for r in rows if r["key"] in ("beutelspacher-2014", "beutelspacher-2022")]
    assert len(year_rows) == 2
    # Keine Jahrgangs-Gruppe erbt die Akzeptanz des jahrlosen Runs
    assert all(r["accept"] is None for r in year_rows), f"Jahrgangs-Zeile hat Run gekapert: {year_rows}"
    # Der jahrlose Run erscheint als eigene Routing-only-Zeile mit Akzeptanz
    own = [r for r in rows if r["key"] == "beutelspacher" and r["accept"] is not None]
    assert len(own) == 1, f"Jahrloser Run nicht als eigene Zeile: {[r['key'] for r in rows]}"
    assert own[0]["routing_only"] is True


def test_run_assignment_is_order_independent():
    """(2, Determinismus) Die Zuordnung darf nicht von der quality_rows-
    Reihenfolge (max(key=len)-Tie) abhängen."""
    quality_rows, all_log_runs = _beutelspacher_two_years()
    a = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    b = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, list(reversed(quality_rows)), current_version=_CUR)
    accept_a = {r["key"]: r["accept"] for r in a}
    accept_b = {r["key"]: r["accept"] for r in b}
    assert accept_a == accept_b, f"order-abhängig: {accept_a} != {accept_b}"


# ── (4) Distinct-Note-Zähler ist SSoT für Kachel UND Sparkline ──────────────
def test_distinct_notes_helper_counts_unique():
    """(4) `_distinct_notes` zählt distinct note_path, Fallback Zeilenindex."""
    rows = [
        {"note_path": "a"},
        {"note_path": "a"},  # Re-Eval derselben Note
        {"note_path": "b"},
        {"note": "c"},  # anderer Identifier-Key
        {},  # ohne Identifier → zählt einzeln (Fallback Index)
    ]
    assert _distinct_notes(rows) == 4  # {a, b, c, <index 4>}


# ── (5) Reine Orphan-Zeilen sichtbar kennzeichnen ───────────────────────────
def test_orphan_only_group_is_flagged():
    """(5) Sind ALLE Versionen einer Quelle > Config-Version, fällt die Zeile
    auf die Orphan-Version zurück (versions[-1]) und wird als `orphan` markiert."""
    quality_rows = [
        _q("Rieder - 2099 - Zukunft.pdf", "r1", "v9.9.99", 0.05, 0.80),
        _q("Rieder - 2099 - Zukunft.pdf", "r2", "v9.9.98", 0.06, 0.70),
    ]
    all_log_runs = [_run("rieder", "Rieder", "v9.9.99", 2, 1)]
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    row = next(r for r in rows if r["key"].startswith("rieder"))
    assert row["orphan"] is True, "reine Orphan-Zeile nicht markiert"
    assert row["version"] == "v9.9.99"  # Fallback versions[-1]


def test_normal_group_is_not_flagged_orphan():
    """(5, Gegenprobe) Eine Zeile mit gültiger (gekappter) Version ist kein Orphan."""
    quality_rows = [_q("Afzal - 2017 - X.pdf", "a1", "v0.3.140", 0.05, 0.80)]
    all_log_runs = [_run("afzal", "Afzal", "v0.3.140", 2, 2)]
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    row = next(r for r in rows if r["key"].startswith("afzal"))
    assert row["orphan"] is False


# ── (6) Leerer Gruppen-Key erzeugt keine Geisterzeile / matcht nicht ALLES ──
def test_empty_group_key_does_not_wildcard_match():
    """(6) Eine Eval-Zeile mit leerem Gruppen-Key (pdf='.pdf') darf per
    `_pdf_matches('')` nicht ALLE Routing-Runs an sich ziehen."""
    quality_rows = [
        _q(".pdf", "ghost", "v0.3.140", 0.05, 0.80),  # leerer Gruppen-Key
        _q("Kling - 2020 - Y.pdf", "k1", "v0.3.140", 0.04, 0.90),
    ]
    all_log_runs = [_run("kling", "Kling", "v0.3.140", 3, 3)]
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    kling = next(r for r in rows if r["key"].startswith("kling"))
    assert kling["accept"] is not None, "Kling-Run wurde von leerem Gruppen-Key geschluckt"
    assert kling["accept_n"] == 1
    # Kein Gruppen-Key ist der leere String
    assert all(r["key"] for r in rows), f"leerer Gruppen-Key als Zeile: {[r['key'] for r in rows]}"


# ── (7) Routing-only-Zeilen als solche gekennzeichnet ───────────────────────
def test_routing_only_flag_set():
    """(7) Routing-only-Zeilen (keine Eval-Daten) tragen routing_only=True,
    Eval-Zeilen routing_only=False."""
    quality_rows = [_q("Nardi - 1996 - Z.pdf", "n1", "v0.3.140", 0.05, 0.80)]
    all_log_runs = [
        _run("nardi", "Nardi", "v0.3.140", 2, 2),
        _run("solo-routing-key", "solo-routing-key", "v0.3.140", 1, 1),  # nur Routing
    ]
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    nardi = next(r for r in rows if r["key"].startswith("nardi"))
    solo = next(r for r in rows if "solo-routing" in r["key"])
    assert nardi["routing_only"] is False
    assert solo["routing_only"] is True
