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
    _pdf_group_key,
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
