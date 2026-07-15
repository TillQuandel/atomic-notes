"""Tests für die Versions×PDF-Paarvergleich-Ansicht (Multi-Perspektiven-Dashboard-
Review 2026-07-15, P1-Empfehlung aller 3 Statistiker, vom adversarialen
Statistiker modifiziert).

Kernbefund (Statistik-Review, PR #293): gepoolte Versions-Deltas über den
vollen PDF-Corpus sind Artefakte, sobald sich der Corpus zwischen zwei
Versionen aendert (v0.3.140 vs v0.3.143 teilten nur 3 von 9/5 PDFs). #293 hat
das fuer das EINE Sparkline-Delta (neueste vs. letzte belastbare Vorversion)
gefixt (`version_delta`, Corpus-Overlap-Guard). Diese Ansicht erweitert das
Prinzip auf eine frei waehlbare Versions×PDF-Matrix:

  1. Deltas NUR auf Schnittmengen (PDFs, die in BEIDEN verglichenen Versionen
     vorkommen) -- `_version_pair_compare`.
  2. Note-gepaart wo moeglich (echte `_note_key`-Uebereinstimmung ueber beide
     Versionen), sonst PDF-Ebene mit Kennzeichnung (`paired`-Flag).
  3. n je Zelle PFLICHT, dedupliziert per `_dedup_latest_per_note` (#293) --
     keine Pseudoreplikation. n<3 wird vom Client gedimmt, nicht versteckt
     (reine UI-Schwelle auf dem `n`-Feld, kein Backend-Flag noetig).
  4. Bestehender Archiv-Mechanismus (`pipeline_runs_archive` /
     `query_archived_pipeline_versions`, #193) schliesst ganze Versionen aus
     -- Aufrufer-Pflicht (Server), NICHT Teil der reinen Aggregationsfunktionen
     hier. Einzelne Ausreisser-Zeilen (z. B. der bekannte, NICHT
     quarantaenierte #232-Artefakt-Lauf) werden NICHT hartkodiert entfernt --
     sie fliessen ein, ihre Dominanz wird ueber n/Min/Max sichtbar statt
     versteckt.
  5. eval_version-Konsistenz ist Aufrufer-Pflicht (dieselbe Konvention wie
     `_dedup_latest_per_note`): `quality_rows` muss vorher auf eine einzelne
     eval_version eingeschraenkt sein.

Alle Fixtures sind rein synthetisch; keine .cache-Daten werden gelesen.
"""

from __future__ import annotations

from generative.eval_dashboard import (
    _calc_version_pdf_matrix,
    _matrix_cell_stats,
    _version_pair_compare,
    build_version_pdf_matrix,
)


def _q(pdf, note, ver, hall=None, cov=None, ts="2026-01-01T00:00:00"):
    """Minimale note_evals-Zeile (Fixture-Konvention wie test_dashboard_pdf_table.py)."""
    return {
        "pdf": pdf,
        "note_path": note,
        "pipeline_version": ver,
        "hallucination_rate": hall,
        "coverage_factual": cov,
        "timestamp": ts,
    }


# ── _matrix_cell_stats ──────────────────────────────────────────────────────


def test_cell_stats_empty_rows_is_none():
    assert _matrix_cell_stats([]) is None


def test_cell_stats_dedups_re_evals_per_note():
    # Dieselbe Note zweimal evaluiert (Re-Eval) -- nur die neueste Zeile zaehlt
    # (#293-Konvention), n=1 nicht n=2.
    rows = [
        _q("a.pdf", "note1", "v1", hall=0.20, cov=0.50, ts="2026-01-01T00:00:00"),
        _q("a.pdf", "note1", "v1", hall=0.10, cov=0.70, ts="2026-01-02T00:00:00"),  # neuere Zeile gewinnt
    ]
    stats = _matrix_cell_stats(rows)
    assert stats["n"] == 1
    assert stats["median_hall"] == 10.0
    assert stats["median_cov"] == 70.0


def test_cell_stats_median_and_min_max_spread():
    rows = [
        _q("a.pdf", "n1", "v1", hall=0.05, cov=0.80),
        _q("a.pdf", "n2", "v1", hall=0.10, cov=0.60),
        _q("a.pdf", "n3", "v1", hall=0.429, cov=0.40),  # Ausreisser (Artefakt-Muster)
    ]
    stats = _matrix_cell_stats(rows)
    assert stats["n"] == 3
    assert stats["median_hall"] == 10.0
    assert stats["min_hall"] == 5.0
    assert stats["max_hall"] == 42.9
    assert stats["median_cov"] == 60.0
    assert stats["min_cov"] == 40.0
    assert stats["max_cov"] == 80.0


def test_cell_stats_single_outlier_row_not_hidden():
    """Requirement 4: eine dominante Einzelzeile (Artefakt-Muster wie der
    #232-Hrastinski-Lauf) wird NICHT hartkodiert ausgeschlossen -- sie bleibt
    die Zelle, n=1 macht ihre Alleinherrschaft ueber den Median sichtbar."""
    rows = [_q("hrastinski.pdf", "only-note", "v1", hall=0.429, cov=0.30)]
    stats = _matrix_cell_stats(rows)
    assert stats["n"] == 1
    assert stats["median_hall"] == 42.9
    assert stats["min_hall"] == stats["max_hall"] == 42.9


def test_cell_stats_rows_without_valid_metric_still_report_n():
    # Notes existieren (n>0), aber kein gueltiger Metrikwert -- Zelle ist nicht
    # None (es gab Daten), median_hall/cov sind None ("--" im Client).
    rows = [_q("a.pdf", "n1", "v1", hall=None, cov=None)]
    stats = _matrix_cell_stats(rows)
    assert stats["n"] == 1
    assert stats["median_hall"] is None
    assert stats["median_cov"] is None


def test_cell_stats_negative_sentinel_excluded():
    # -1.0 ist der bestehende Sentinel fuer "ungueltig" (vgl. _chart_scatter).
    rows = [_q("a.pdf", "n1", "v1", hall=-1.0, cov=-1.0)]
    stats = _matrix_cell_stats(rows)
    assert stats["n"] == 1
    assert stats["median_hall"] is None
    assert stats["median_cov"] is None


def test_cell_stats_zero_coverage_factual_not_swallowed_by_or_fallback():
    """D4-Anti-Pattern-Regression (Dashboard-Review 2026-07-15): eine ECHTE
    coverage_factual=0.0 ist falsy — ein `or`-Fallback wuerde sie verschlucken
    und faelschlich coverage_rate nehmen. 0%-Coverage-Zeilen existieren real
    (Jockisch-Faelle). coverage_factual=0.0 MUSS als 0.0 einfliessen."""
    rows = [
        {
            "pdf": "a.pdf",
            "note_path": "n1",
            "pipeline_version": "v1",
            "hallucination_rate": 0.1,
            "coverage_factual": 0.0,
            "coverage_rate": 0.8,  # darf NICHT gewinnen
            "timestamp": "2026-01-01T00:00:00",
        }
    ]
    stats = _matrix_cell_stats(rows)
    assert stats["median_cov"] == 0.0  # nicht 80.0
    assert stats["min_cov"] == 0.0
    assert stats["max_cov"] == 0.0


def test_cell_stats_zero_coverage_factual_without_coverage_rate_counts_as_zero():
    # coverage_factual=0.0 + coverage_rate=None: `or` wuerde die Zeile komplett
    # verwerfen (None) — sie muss als 0.0 zaehlen.
    rows = [
        {
            "pdf": "a.pdf",
            "note_path": "n1",
            "pipeline_version": "v1",
            "hallucination_rate": 0.1,
            "coverage_factual": 0.0,
            "coverage_rate": None,
            "timestamp": "2026-01-01T00:00:00",
        }
    ]
    stats = _matrix_cell_stats(rows)
    assert stats["median_cov"] == 0.0  # nicht None/verworfen


def test_cell_stats_none_coverage_factual_falls_back_to_coverage_rate():
    # Der legitime Fallback-Fall bleibt erhalten: NUR bei coverage_factual=None
    # greift coverage_rate (v4-Zeilen schreiben coverage_factual immer NULL, #233).
    rows = [
        {
            "pdf": "a.pdf",
            "note_path": "n1",
            "pipeline_version": "v1",
            "hallucination_rate": 0.1,
            "coverage_factual": None,
            "coverage_rate": 0.8,
            "timestamp": "2026-01-01T00:00:00",
        }
    ]
    stats = _matrix_cell_stats(rows)
    assert stats["median_cov"] == 80.0


def test_pair_compare_zero_coverage_factual_not_swallowed_by_or_fallback():
    """Dieselbe D4-Regression fuer den Paarvergleichs-Pfad (_pair_metric_stats):
    cov_a muss die echte 0.0 aus coverage_factual zeigen, nicht coverage_rate."""

    def _row(note, ver, cf, cr):
        return {
            "pdf": "Shared.pdf",
            "note_path": note,
            "pipeline_version": ver,
            "hallucination_rate": 0.1,
            "coverage_factual": cf,
            "coverage_rate": cr,
            "timestamp": "2026-01-01T00:00:00",
        }

    rows = [
        _row("a1", "vA", 0.0, 0.8),  # echte 0%-Coverage, coverage_rate darf nicht gewinnen
        _row("b1", "vB", 0.0, None),  # echte 0%-Coverage, darf nicht verworfen werden
    ]
    cmp = _version_pair_compare(rows, "vA", "vB")
    cell = cmp["per_pdf"]["shared"]
    assert cell["cov_a"] == 0.0  # nicht 80.0
    assert cell["cov_b"] == 0.0  # nicht None
    assert cell["cov_delta"] == 0.0


# ── _calc_version_pdf_matrix ────────────────────────────────────────────────


def test_matrix_places_cells_by_pdf_and_version():
    rows = [
        _q("Bates - 2017 - Information Behavior.pdf", "b1", "v0.3.10", hall=0.05, cov=0.80),
        _q("Bates - 2017 - Information Behavior.pdf", "b2", "v0.3.20", hall=0.15, cov=0.60),
        _q("Schlebbe - 2022 - Info Need.pdf", "s1", "v0.3.20", hall=0.20, cov=0.50),
    ]
    m = _calc_version_pdf_matrix(rows, versions=["v0.3.10", "v0.3.20"])
    assert m["versions"] == ["v0.3.10", "v0.3.20"]
    # _pdf_group_key = Autor-Jahr-Slug (SSoT mit dem bestehenden PDF-Filter,
    # #202/#194) -- NICHT der volle Titel-Slug.
    keys = {p["key"] for p in m["pdfs"]}
    assert keys == {"bates-2017", "schlebbe-2022"}
    bates_key = "bates-2017"
    assert m["cells"][bates_key]["v0.3.10"]["n"] == 1
    assert m["cells"][bates_key]["v0.3.10"]["median_hall"] == 5.0
    # Kein Bates-Eval in v0.3.20? Doch (b2) -- pruefen:
    assert m["cells"][bates_key]["v0.3.20"]["median_hall"] == 15.0
    # Schlebbe hat KEINE Zeile in v0.3.10 -- Zelle ist None, nicht fehlend/Crash.
    schlebbe_key = "schlebbe-2022"
    assert m["cells"][schlebbe_key]["v0.3.10"] is None
    assert m["cells"][schlebbe_key]["v0.3.20"]["n"] == 1


def test_matrix_pdf_label_fallback_to_short_name_when_not_in_pdf_labels():
    rows = [_q("Unbekannter Autor - 2020 - Ein Titel.pdf", "n1", "v1", hall=0.1, cov=0.5)]
    m = _calc_version_pdf_matrix(rows, versions=["v1"])
    assert m["pdfs"][0]["label"] == "Unbekannter Autor (2020)"


def test_matrix_auto_version_selection_reuses_top_versions_capping():
    # Ohne expliziten versions-Parameter: automatische Version-Auswahl nutzt
    # dieselbe "15 neueste mit n>=3, neueste immer dabei"-Regel wie das
    # bestehende Versions-Dropdown (_top_versions, SSoT) -- hier verifiziert
    # ueber eine Version mit nur 1 Zeile (< min_n), die trotzdem die NEUESTE
    # ist und deshalb erscheinen muss; eine aeltere robuste Version (n=3)
    # bleibt ebenfalls drin.
    rows = [_q("a.pdf", f"n{i}", "v0.3.1", hall=0.1, cov=0.5) for i in range(3)]
    rows.append(_q("a.pdf", "newest-note", "v0.3.9", hall=0.2, cov=0.5))  # n=1, aber neueste
    m = _calc_version_pdf_matrix(rows)
    assert m["versions"] == ["v0.3.1", "v0.3.9"]


def test_matrix_versions_ascending_oldest_to_newest():
    # Je >=3 Zeilen (min_n-Schwelle von `_top_versions`), sonst faellt die
    # aeltere Version durch den Einzel-Note-Wegwerflauf-Filter.
    rows = [_q("a.pdf", f"n{i}", "v0.3.20", hall=0.1, cov=0.5) for i in range(3)]
    rows += [_q("a.pdf", f"m{i}", "v0.3.10", hall=0.1, cov=0.5) for i in range(3)]
    m = _calc_version_pdf_matrix(rows)
    assert m["versions"] == ["v0.3.10", "v0.3.20"]


# ── _version_pair_compare: Schnittmengen-Regel (Requirement 1) ─────────────


def test_pair_compare_common_and_only_split():
    rows = [
        _q("Shared.pdf", "s1", "vA", hall=0.10, cov=0.60),
        _q("OnlyA.pdf", "a1", "vA", hall=0.05, cov=0.90),
        _q("Shared.pdf", "s2", "vB", hall=0.30, cov=0.40),
        _q("OnlyB.pdf", "b1", "vB", hall=0.50, cov=0.20),
    ]
    cmp = _version_pair_compare(rows, "vA", "vB")
    assert cmp["common_pdfs"] == ["shared"]
    assert cmp["only_a"] == ["onlya"]
    assert cmp["only_b"] == ["onlyb"]


def test_pair_compare_overall_delta_uses_only_common_pdfs_not_full_pool():
    """Requirement 1, direkte Regression: ein gepooltes Delta ueber ALLE Notes
    (inkl. der Nicht-gemeinsamen PDFs) waere hier stark negativ (0 %-Notes
    dominieren vA), das Schnittmengen-Delta ist dagegen klar positiv (30%-10%).
    Der Test faellt durch, wenn `overall` faelschlich ueber den vollen Pool
    statt nur `common_pdfs` rechnet."""
    rows = [
        _q("Shared.pdf", "s1", "vA", hall=0.10, cov=0.60),
        # vA hat viele zusaetzliche 0%-Notes aus einer PDF, die vB nicht teilt --
        # wuerden sie mitgepoolt, zoege das vA-Gesamt Richtung 0 % und das Delta
        # kippt implausibel stark positiv.
        *[_q("OnlyA.pdf", f"a{i}", "vA", hall=0.0, cov=0.95) for i in range(20)],
        _q("Shared.pdf", "s2", "vB", hall=0.30, cov=0.40),
    ]
    cmp = _version_pair_compare(rows, "vA", "vB")
    assert cmp["overall"]["hall_a"] == 10.0  # NUR Shared.pdf, nicht der volle vA-Pool
    assert cmp["overall"]["hall_b"] == 30.0
    assert cmp["overall"]["hall_delta"] == 20.0
    assert cmp["overall"]["n_common_pdfs"] == 1
    assert cmp["overall"]["n_pdfs_a"] == 2
    assert cmp["overall"]["n_pdfs_b"] == 1


def test_pair_compare_no_common_pdfs_yields_no_overall_delta():
    rows = [
        _q("OnlyA.pdf", "a1", "vA", hall=0.1, cov=0.5),
        _q("OnlyB.pdf", "b1", "vB", hall=0.2, cov=0.5),
    ]
    cmp = _version_pair_compare(rows, "vA", "vB")
    assert cmp["common_pdfs"] == []
    assert cmp["overall"]["hall_delta"] is None
    assert cmp["per_pdf"] == {}


# ── _version_pair_compare: Note-Paarung (Requirement 2) ────────────────────


def test_pair_compare_pairs_by_shared_note_key_when_available():
    # Dieselbe Note (gleicher note_path, nur Namespace-Prefix wechselt --
    # #293-_note_key-Normalisierung) taucht in beiden Versionen auf.
    rows = [
        _q("Shared.pdf", "vault__stable-note.md", "vA", hall=0.10, cov=0.70),
        _q("Shared.pdf", "unpaired-a.md", "vA", hall=0.90, cov=0.10),
        _q("Shared.pdf", "inbox__stable-note.md", "vB", hall=0.20, cov=0.60),
        _q("Shared.pdf", "unpaired-b.md", "vB", hall=0.05, cov=0.95),
    ]
    cmp = _version_pair_compare(rows, "vA", "vB")
    cell = cmp["per_pdf"]["shared"]
    assert cell["paired"] is True
    assert cell["n_paired"] == 1
    # Nur die gemeinsame Note fliesst ein -- nicht unpaired-a/-b.
    assert cell["hall_a"] == 10.0
    assert cell["hall_b"] == 20.0
    assert cell["hall_delta"] == 10.0


def test_pair_compare_falls_back_to_pdf_level_without_shared_notes():
    rows = [
        _q("Shared.pdf", "a1", "vA", hall=0.10, cov=0.60),
        _q("Shared.pdf", "a2", "vA", hall=0.20, cov=0.50),
        _q("Shared.pdf", "b1", "vB", hall=0.30, cov=0.40),
    ]
    cmp = _version_pair_compare(rows, "vA", "vB")
    cell = cmp["per_pdf"]["shared"]
    assert cell["paired"] is False
    assert cell["n_paired"] is None
    assert cell["n_a"] == 2  # alle deduplizierten Notes der PDF (PDF-Ebene)
    assert cell["n_b"] == 1
    assert cell["hall_a"] == 15.0  # Median aus 10/20


def test_pair_compare_rows_without_note_identifier_never_falsely_pair():
    """Regression: `_note_key`s Index-Fallback fuer Zeilen OHNE Identifier
    (`__row0`, `__row1`, ...) darf zwei UNABHAENGIGE Zeilen aus verschiedenen
    Versionen nicht ueber zufaellig gleiche Listenposition als 'gepaart'
    erkennen -- das waere eine Scheinpaarung ohne echte Note-Identitaet."""
    rows = [
        {"pdf": "Shared.pdf", "pipeline_version": "vA", "hallucination_rate": 0.10, "coverage_factual": 0.5},
        {"pdf": "Shared.pdf", "pipeline_version": "vB", "hallucination_rate": 0.90, "coverage_factual": 0.1},
    ]
    cmp = _version_pair_compare(rows, "vA", "vB")
    cell = cmp["per_pdf"]["shared"]
    assert cell["paired"] is False


# ── Produktionsmuster #293 (v0.3.140 vs v0.3.143, 3 von 9/5 PDFs geteilt) ──


def _prod_pattern_rows():
    # Nachbau des #293-Produktionsbelegs (test_dashboard_delta_pdf_overlap.py):
    # v0.3.140 hat 9 PDF-Quellen, v0.3.143 hat 5, nur 3 werden geteilt.
    rows = []
    v140 = {
        "assfalg-2013": 2,
        "ebner-und-gegenfurtner-2019": 7,
        "hrastinski-2008": 6,
        "knowles-from-pedagogy-to-andragogy": 9,
        "mahmood-und-university-of-the-punjab-2016": 4,
    }
    v143 = {
        "bates-information-behavior": 4,
        "hrastinski-2008": 2,
        "schlebbe-und-greifeneder-2022": 2,
    }
    for pdf, n in v140.items():
        for i in range(n):
            rows.append(_q(f"{pdf}.pdf", f"{pdf}-140-{i}", "v0.3.140", hall=0.05, cov=0.8))
    for pdf, n in v143.items():
        for i in range(n):
            rows.append(_q(f"{pdf}.pdf", f"{pdf}-143-{i}", "v0.3.143", hall=0.30, cov=0.4))
    return rows


def test_prod_pattern_140_143_only_hrastinski_shared():
    cmp = _version_pair_compare(_prod_pattern_rows(), "v0.3.140", "v0.3.143")
    assert cmp["common_pdfs"] == ["hrastinski-2008"]
    assert cmp["overall"]["n_common_pdfs"] == 1
    assert cmp["overall"]["n_pdfs_a"] == 5
    assert cmp["overall"]["n_pdfs_b"] == 3
    # Deltas beziehen sich NUR auf hrastinski-2008 (6 Notes @5% vs 2 Notes @30%),
    # nicht auf den vollen (weitgehend ausgetauschten) Corpus.
    assert cmp["overall"]["hall_a"] == 5.0
    assert cmp["overall"]["hall_b"] == 30.0


# ── build_version_pdf_matrix: Orchestrierung Matrix + alle Versionspaare ──


def test_build_version_pdf_matrix_includes_compare_for_every_version_pair():
    # Je >=3 Zeilen (min_n-Schwelle von `_top_versions`), sonst wuerden die
    # aelteren Versionen als Einzel-Note-Wegwerflaeufe herausgefiltert.
    rows = [_q("a.pdf", f"n1-{i}", "v1", hall=0.1, cov=0.5) for i in range(3)]
    rows += [_q("a.pdf", f"n2-{i}", "v2", hall=0.2, cov=0.5) for i in range(3)]
    rows += [_q("a.pdf", f"n3-{i}", "v3", hall=0.3, cov=0.5) for i in range(3)]
    m = build_version_pdf_matrix(rows)
    assert set(m["compare"].keys()) == {"v1|v2", "v1|v3", "v2|v3"}
    assert m["compare"]["v1|v3"]["version_a"] == "v1"
    assert m["compare"]["v1|v3"]["version_b"] == "v3"


# ── Server-Integration: eval_dashboard_server.build_data() liefert pair_matrix ──


def _dbrow(note, ver, pdf, hall, ts, eval_version="4.1", cov=0.5, run_id=None):
    """Minimale note_evals-DB-Zeile (Konvention wie test_dashboard_delta_pdf_overlap.py:_eval)."""
    return {
        "run_id": run_id or f"r-{note}",
        "note_path": note,
        "pipeline_version": ver,
        "version": ver,
        "hallucination_rate": hall,
        "anchors_total": 10,
        "anchors_hallucinated": 0,
        "coverage_factual": cov,
        "pdf": pdf,
        "eval_version": eval_version,
        "timestamp": ts,
    }


def _patched_build_data(monkeypatch, evals, current_version="v0.3.143", archived=None, **kwargs):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    monkeypatch.setattr(_cfg, "AGENT_VERSION", current_version)
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: evals)
    monkeypatch.setattr(_gdb, "query_archived_pipeline_versions", lambda *a, **k: archived or [])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])
    return S.build_data(**kwargs)


def _matrix_rows_for(n, ver, pdf, hall, eval_version="4.1"):
    return [_dbrow(f"{pdf}-{ver}-{i}", ver, pdf, hall, f"2026-06-01T00:00:{i:02d}", eval_version) for i in range(n)]


def test_build_data_includes_pair_matrix_with_versions_pdfs_cells_compare(monkeypatch):
    evals = _matrix_rows_for(3, "v0.3.142", "Shared.pdf", 0.1) + _matrix_rows_for(3, "v0.3.143", "Shared.pdf", 0.2)
    data = _patched_build_data(monkeypatch, evals)
    pm = data["pair_matrix"]
    assert pm["versions"] == ["v0.3.142", "v0.3.143"]
    assert {p["key"] for p in pm["pdfs"]} == {"shared"}
    assert pm["cells"]["shared"]["v0.3.142"]["n"] == 3
    assert "v0.3.142|v0.3.143" in pm["compare"]
    assert pm["eval_version"] == "4.1"


def test_pair_matrix_pools_only_current_eval_version_not_legacy_1_3(monkeypatch):
    """Requirement #5: 1.3- und 4.1-Zeilen duerfen nie zusammen gepoolt werden --
    ein Mix waere ein Messartefakt. build_data() waehlt default die neueste
    eval_version (4.1); die Matrix darf die 1.3-Zeilen dieser Version/PDF NICHT
    mitzaehlen (n muesste sonst 6 statt 3 sein)."""
    evals = _matrix_rows_for(3, "v0.3.143", "Shared.pdf", 0.1, eval_version="1.3")
    evals += _matrix_rows_for(3, "v0.3.143", "Shared.pdf", 0.2, eval_version="4.1")
    data = _patched_build_data(monkeypatch, evals)
    pm = data["pair_matrix"]
    assert pm["eval_version"] == "4.1"
    assert pm["cells"]["shared"]["v0.3.143"]["n"] == 3
    assert pm["cells"]["shared"]["v0.3.143"]["median_hall"] == 20.0


def test_pair_matrix_excludes_archived_versions_and_reports_them(monkeypatch):
    """Requirement #4: bestehender Archiv-Mechanismus (`pipeline_runs_archive`,
    #193) schliesst verwaiste WIP-Versionen aus der Matrix aus -- UND wird
    dabei kenntlich gemacht (`excluded_archived_versions`), statt sie
    stillschweigend verschwinden zu lassen."""
    evals = _matrix_rows_for(3, "v0.3.141", "Shared.pdf", 0.9)  # archivierte WIP-Version
    evals += _matrix_rows_for(3, "v0.3.143", "Shared.pdf", 0.1)
    data = _patched_build_data(monkeypatch, evals, archived=["v0.3.141"])
    pm = data["pair_matrix"]
    assert "v0.3.141" not in pm["versions"]
    assert pm["excluded_archived_versions"] == ["v0.3.141"]
    assert "v0.3.143" in pm["versions"]


def test_pair_matrix_ignores_active_single_value_pdf_and_version_filters(monkeypatch):
    """Design-Entscheid: die Matrix zeigt bewusst IMMER den vollen Corpus der
    aktiven eval_version -- unabhaengig von den bestehenden Einzelwert-Filtern
    (PDF/Pipeline-Version in der Sidebar), sonst kollabiert sie auf 1 Zelle und
    verfehlt ihren Zweck (Versions×PDF-UEBERSICHT). Ein aktiver `pdf`- oder
    `pipeline_version`-Query-Parameter darf die Matrix nicht einschraenken."""
    evals = _matrix_rows_for(3, "v0.3.142", "Shared.pdf", 0.1)
    evals += _matrix_rows_for(3, "v0.3.143", "Other.pdf", 0.2)
    data = _patched_build_data(monkeypatch, evals, pipeline_version="v0.3.142")
    pm = data["pair_matrix"]
    assert set(pm["versions"]) == {"v0.3.142", "v0.3.143"}
    assert {p["key"] for p in pm["pdfs"]} == {"shared", "other"}


def test_pair_matrix_does_not_hardcode_exclude_known_artifact_row(monkeypatch):
    """Requirement #4 (Till-Entscheid): die bekannte, NICHT quarantaenierte
    #232-Artefakt-Zeile darf nicht hartkodiert herausgefiltert werden -- ein
    dominanter Ausreisser (hier stellvertretend: eine einzelne 90%-Zeile in
    einer sonst sauberen PDF-Version) muss in der Matrix sichtbar bleiben,
    inkl. n=1 als Signal fuer die Alleinherrschaft ueber den Zellwert."""
    evals = _matrix_rows_for(3, "v0.3.142", "Bates.pdf", 0.05)
    evals += [_dbrow("artefakt-note", "v0.3.143", "Bates.pdf", 0.90, "2026-07-12T21:51:18")]
    data = _patched_build_data(monkeypatch, evals)
    pm = data["pair_matrix"]
    cell = pm["cells"]["bates"]["v0.3.143"]
    assert cell is not None
    assert cell["n"] == 1
    assert cell["median_hall"] == 90.0
