"""Tests für Repeat-Sweep-Pooling (Till-Go 2026-07-16, Trendfähigkeits-Studie).

Kontext: Seit 2026-07-16 existieren echte Eval-Wiederholungsgruppen (Baseline-
Korpus 3× unter eval_version 4.3 gemessen: 1× `reeval_baseline.py`
(pipeline_runs.pipeline_version="reeval"), 2× `--repeat`
(pipeline_version="reeval-repeat"), JSONL-Feld `repeat_sweep: true`). DB-Beleg
(atomic_analytics.db, read-only Abfrage 2026-07-16): 81 note_evals-Zeilen /
27 distinct Notes bei eval_version=4.3, je Note EXAKT 3 Zeilen (eine je
run_id: 20260715-233052 "reeval" 14/400 = 3,5 %, 20260716-151038
"reeval-repeat" 10/400 = 2,5 %, 20260716-165018 "reeval-repeat" 16/400 = 4,0 %).
Anker-Pool über alle 3 Sweeps: 40/1200 = 3,33 % (gerundet 3,3 %, 1 Nachkomma-
stelle wie überall im Dashboard, `_pooled_hall_stats`).

Bug (bisher): `_dedup_latest_per_note` nahm pro Note die neueste Zeile (hier:
den jeweils letzten Sweep) — die KPI-Kachel zeigte dadurch nach jedem neuen
`--repeat`-Lauf einen ANDEREN Wert (zuletzt 16/400 = 4,0 %) statt des stabilen
Referenzwerts 3,3 %, weil 2 von 3 echten Messungen je Note stillschweigend
verworfen wurden.

Fix: `_dedup_latest_per_note(rows, pipeline_runs=...)` erkennt pro Note Zeilen,
deren `run_id` an einen `pipeline_runs`-Eintrag mit `pipeline_version` in
`_REEVAL_PIPELINE_VERSIONS` ("reeval"/"reeval-repeat") hängt, und poolt sie
(`_pool_repeat_rows`) zu EINER synthetischen Zeile, BEVOR der bestehende
Latest-Wins-Vergleich (#293-Vertrag) greift. Ohne `pipeline_runs` (Default
None) bleibt das Verhalten byte-identisch zu vorher.
"""

from __future__ import annotations

from generative.eval_dashboard import (
    _calc_kpis,
    _calc_pdf_table,
    _calc_version_pdf_matrix,
    _dedup_latest_per_note,
    _matrix_cell_stats,
    _pool_repeat_rows,
    _pooled_hall_pct,
    _version_pair_compare,
    build_version_pdf_matrix,
)

REEVAL_RUNS = [
    {"run_id": "reeval-run", "pipeline_version": "reeval"},
    {"run_id": "repeat-run-1", "pipeline_version": "reeval-repeat"},
    {"run_id": "repeat-run-2", "pipeline_version": "reeval-repeat"},
]


def _qrow(note, run_id, hall, total, hallucinated, ts, cov=0.5, eval_id=None):
    return {
        "note_path": note,
        "run_id": run_id,
        "version": "v0.3.144",
        "hallucination_rate": hall,
        "anchors_total": total,
        "anchors_hallucinated": hallucinated,
        "coverage_factual": cov,
        "timestamp": ts,
        "eval_id": eval_id or f"{ts}__{note}",
    }


# ── _pool_repeat_rows: Pool-Mathematik am echten Zahlenbeispiel ────────────
# (DB-Beleg 2026-07-16: eval_version 4.3, 27 Notes × 3 Sweeps, hier auf EINE
# Note reduziert nachgebaut -- die Summenmathematik ist identisch, ob 400
# Anker aus 27 Notes oder aus einer einzigen stammen.)


def test_pool_repeat_rows_anchor_weighted_matches_production_example():
    rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
        _qrow("baseline.md", "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
    ]
    pooled = _pool_repeat_rows(rows)
    assert pooled["anchors_total"] == 1200
    assert pooled["anchors_hallucinated"] == 40
    assert round(pooled["hallucination_rate"] * 100, 1) == 3.3  # 40/1200 = 3,33... -> 3,3 (1 Nachkommastelle SSoT)
    assert pooled["n_measurements"] == 3


def test_pool_repeat_rows_hall_min_max_is_spread_of_individual_measurements():
    rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
        _qrow("baseline.md", "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
    ]
    pooled = _pool_repeat_rows(rows)
    assert pooled["hall_min"] == 2.5
    assert pooled["hall_max"] == 4.0


def test_pool_repeat_rows_marks_repeat_pooled_and_inherits_latest_identity():
    rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
    ]
    pooled = _pool_repeat_rows(rows)
    assert pooled["repeat_pooled"] is True
    assert pooled["n_measurements"] == 2
    # Identitaets-/Metafelder von der zeitlich juengsten Zeile
    assert pooled["timestamp"] == "2026-07-16T16:50:18"
    assert pooled["run_id"] == "repeat-run-2"
    assert pooled["note_path"] == "baseline.md"


def test_pool_repeat_rows_coverage_is_mean_of_valid_values_no_anchor_pool():
    """note_evals hat keine Roh-Counts fuer Coverage (kein claims_total/
    claims_supported in der DB-Zeile) -- notengewichtetes Mittel, dieselbe
    Einschraenkung wie `_version_pair_compare`s `_cov_mean`."""
    rows = [
        _qrow("baseline.md", "reeval-run", 0.0, 10, 0, "2026-07-15T23:30:52", cov=0.2),
        _qrow("baseline.md", "repeat-run-1", 0.0, 10, 0, "2026-07-16T15:10:38", cov=0.8),
    ]
    pooled = _pool_repeat_rows(rows)
    assert pooled["coverage_factual"] == 0.5
    assert pooled["coverage_rate"] == 0.5


def test_pool_repeat_rows_coverage_excludes_negative_sentinel():
    rows = [
        _qrow("baseline.md", "reeval-run", 0.0, 10, 0, "2026-07-15T23:30:52", cov=-1.0),
        _qrow("baseline.md", "repeat-run-1", 0.0, 10, 0, "2026-07-16T15:10:38", cov=0.6),
    ]
    pooled = _pool_repeat_rows(rows)
    assert pooled["coverage_factual"] == 0.6  # nur die gueltige Zeile zaehlt


# ── _dedup_latest_per_note: pipeline_runs-Parameter ────────────────────────


def test_dedup_without_pipeline_runs_param_is_byte_identical_to_before():
    """Ohne `pipeline_runs` (Default None) bleibt das Verhalten unveraendert --
    keine Zeile kann ohne diese Zusatzinfo als Re-Eval-Familie erkannt werden."""
    rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
        _qrow("baseline.md", "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
    ]
    out = _dedup_latest_per_note(rows)
    assert len(out) == 1
    assert out[0]["hallucination_rate"] == 0.04  # neueste Zeile, nicht gepoolt
    assert "repeat_pooled" not in out[0]


def test_dedup_with_pipeline_runs_pools_reeval_family_group():
    rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
        _qrow("baseline.md", "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
    ]
    out = _dedup_latest_per_note(rows, REEVAL_RUNS)
    assert len(out) == 1
    assert out[0]["repeat_pooled"] is True
    assert out[0]["anchors_total"] == 1200
    assert out[0]["anchors_hallucinated"] == 40
    assert round(out[0]["hallucination_rate"] * 100, 1) == 3.3


def test_dedup_single_reeval_row_not_pooled():
    """Nur EINE Re-Eval-Zeile (kein echtes Wiederholungspaar) -- nichts zu
    poolen, sie nimmt unveraendert am Latest-Wins-Vergleich teil."""
    rows = [_qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52")]
    out = _dedup_latest_per_note(rows, REEVAL_RUNS)
    assert len(out) == 1
    assert "repeat_pooled" not in out[0]
    assert out[0]["hallucination_rate"] == 0.035


# ── Regression: normale Mehrfachzeilen (NICHT Re-Eval-Familie) bleiben #293 ─
# (4.1-artige Duplikate/Re-Evals ausserhalb der Re-Eval-Familie -- der
# #293-Vertrag "neueste gewinnt" bleibt fuer sie bestehen, auch wenn
# `pipeline_runs` explizit uebergeben wird.)


def test_dedup_normal_duplicates_stay_latest_wins_even_with_pipeline_runs_given():
    rows = [
        _qrow("dup.md", "normal-run", 0.0, 17, 0, "2026-06-21T19:50:12"),
        _qrow("dup.md", "normal-run", 0.0, 17, 0, "2026-06-21T21:20:53"),
        _qrow("dup.md", "normal-run", 0.294, 17, 5, "2026-07-05T19:59:18"),  # neueste
    ]
    out_without = _dedup_latest_per_note(rows)
    out_with = _dedup_latest_per_note(rows, REEVAL_RUNS)  # normal-run ist NICHT in REEVAL_RUNS
    assert out_without == out_with
    assert len(out_with) == 1
    assert out_with[0]["hallucination_rate"] == 0.294
    assert "repeat_pooled" not in out_with[0]


def test_pooled_hall_matches_reference_value_not_latest_sweep_only():
    """Direkter Vorher/Nachher-Vergleich am Produktionsmuster: Latest-Wins
    (bisher) liefert 4,0 % (nur der juengste Sweep), Pooling liefert den
    stabilen Referenzwert 3,3 % (Rauschanalyse)."""
    rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
        _qrow("baseline.md", "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
    ]
    latest_only_pct = _pooled_hall_pct(_dedup_latest_per_note(rows))
    pooled_pct = _pooled_hall_pct(_dedup_latest_per_note(rows, REEVAL_RUNS))
    assert latest_only_pct == 4.0
    assert pooled_pct == 3.3


# ── Mischfall: Re-Eval-Gruppe UND normale Pipeline-Zeile derselben Note ────


def test_mixed_case_pooled_group_wins_over_older_normal_row():
    """Note hat 2 Re-Eval-Zeilen (Familie) + 1 aeltere normale Zeile in
    derselben eval_version. Erst poolen (2 Re-Eval-Zeilen -> 1 Pool-Zeile),
    DANN Latest-Wins zwischen Pool-Zeile und der normalen Zeile -- die
    Pool-Zeile ist juenger und gewinnt MIT den gepoolten Werten (nicht nur
    dem juengsten Einzelwert)."""
    rows = [
        _qrow("mix.md", "normal-run", 0.5, 20, 10, "2026-07-09T00:00:00"),  # aelteste
        _qrow("mix.md", "reeval-run", 0.05, 20, 1, "2026-07-10T00:00:00"),
        _qrow("mix.md", "repeat-run-1", 0.10, 20, 2, "2026-07-11T00:00:00"),  # juengste Re-Eval-Zeile
    ]
    out = _dedup_latest_per_note(rows, REEVAL_RUNS)
    assert len(out) == 1
    row = out[0]
    assert row["repeat_pooled"] is True
    assert row["anchors_total"] == 40
    assert row["anchors_hallucinated"] == 3
    assert round(row["hallucination_rate"] * 100, 2) == 7.5  # (1+2)/40, NICHT 10.0 (nur juengste Re-Eval-Zeile)


def test_mixed_case_newer_normal_row_wins_outright_pooling_discarded():
    """Ist die normale Zeile juenger als BEIDE Re-Eval-Zeilen, gewinnt sie
    unveraendert -- die intern berechnete Pool-Zeile wird verworfen (Latest-
    Wins bleibt die entscheidende Regel, Pooling ist nur eine Kandidatin)."""
    rows = [
        _qrow("mix.md", "reeval-run", 0.05, 20, 1, "2026-07-09T00:00:00"),
        _qrow("mix.md", "repeat-run-1", 0.10, 20, 2, "2026-07-10T00:00:00"),
        _qrow("mix.md", "normal-run", 0.5, 20, 10, "2026-07-12T00:00:00"),  # neueste
    ]
    out = _dedup_latest_per_note(rows, REEVAL_RUNS)
    assert len(out) == 1
    row = out[0]
    assert row["hallucination_rate"] == 0.5
    assert "repeat_pooled" not in row


# ── n zaehlt Notes, nie Messungen (Pseudoreplikations-Schutz) ──────────────


def test_calc_kpis_n_notes_counts_notes_not_measurements():
    """3 Notes x 3 Re-Eval-Messungen (9 Zeilen) -> n_notes MUSS 3 zeigen,
    nicht 9 (keine Pseudoreplikation, Analogon zum realen 27/81-Fall)."""
    rows = []
    for note in ("a.md", "b.md", "c.md"):
        rows += [
            _qrow(note, "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
            _qrow(note, "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
            _qrow(note, "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
        ]
    kpis = _calc_kpis({}, [], rows, [], current_version="v0.3.144", pipeline_runs=REEVAL_RUNS)
    assert kpis["n_notes"] == 3
    # gepoolte Fehlerquote: identisch fuer jede der 3 Notes (40/1200 je Note) ->
    # 3,3 % ueber den gesamten Corpus (Summe skaliert linear, Rate bleibt gleich).
    assert kpis["avg_hall"] == 3.3
    assert kpis["hall_anchors_total"] == 3600  # 3 Notes x 1200
    assert kpis["hall_notes_n"] == 3


def test_calc_pdf_table_pools_reeval_family_and_keeps_n_notes_correct():
    quality_rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
        _qrow("baseline.md", "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
    ]
    for r in quality_rows:
        r["pdf"] = "Quelle - 2020 - X.pdf"
        r["pipeline_version"] = "v0.3.144"
    all_log_runs = [
        {
            "key": "quelle-2020",
            "label": "Quelle",
            "ver": "v0.3.144",
            "n_total": 1,
            "n_vault": 1,
            "accept_pct": 100.0,
            "words": None,
        }
    ]
    from generative.eval_dashboard import _build_log_data

    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version="v0.3.144", pipeline_runs=REEVAL_RUNS)
    row = next(r for r in rows if r["key"].startswith("quelle"))
    assert row["n_notes"] == 1
    assert row["hall"] == 3.3


# ── Matrix-Zelle: gepoolte Aggregat-Felder fuer Client-Tooltip ─────────────


def test_matrix_cell_stats_exposes_pooled_aggregate_fields():
    rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
        _qrow("baseline.md", "repeat-run-2", 0.04, 400, 16, "2026-07-16T16:50:18"),
    ]
    stats = _matrix_cell_stats(rows, REEVAL_RUNS)
    assert stats["n"] == 1
    assert stats["n_pooled"] == 1
    assert stats["pooled_n_measurements"] == 3
    assert stats["pooled_hall_min"] == 2.5
    assert stats["pooled_hall_max"] == 4.0


def test_matrix_cell_stats_no_pooled_fields_when_nothing_pooled():
    rows = [_qrow("a.md", "normal-run", 0.1, 10, 1, "2026-01-01T00:00:00")]
    stats = _matrix_cell_stats(rows, REEVAL_RUNS)
    assert stats["n_pooled"] == 0
    assert stats["pooled_n_measurements"] is None
    assert stats["pooled_hall_min"] is None
    assert stats["pooled_hall_max"] is None


def test_matrix_cell_stats_backward_compatible_without_pipeline_runs_arg():
    """Bestehende Aufrufer (Positional, ohne zweites Argument) bleiben gueltig."""
    rows = [_qrow("a.md", "normal-run", 0.1, 10, 1, "2026-01-01T00:00:00")]
    stats = _matrix_cell_stats(rows)
    assert stats["n"] == 1


def test_calc_version_pdf_matrix_threads_pipeline_runs_into_cells():
    rows = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
    ]
    for r in rows:
        r["pdf"] = "a.pdf"
        r["version"] = "v1"
    m = _calc_version_pdf_matrix(rows, versions=["v1"], pipeline_runs=[REEVAL_RUNS[0], REEVAL_RUNS[1]])
    cell = m["cells"]["a"]["v1"]
    assert cell["n"] == 1
    assert cell["n_pooled"] == 1


def test_version_pair_compare_accepts_pipeline_runs_and_pools_each_side():
    rows_a = [
        _qrow("baseline.md", "reeval-run", 0.035, 400, 14, "2026-07-15T23:30:52"),
        _qrow("baseline.md", "repeat-run-1", 0.025, 400, 10, "2026-07-16T15:10:38"),
    ]
    for r in rows_a:
        r["pdf"] = "Shared.pdf"
        r["version"] = "vA"
    rows_b = [_qrow("other.md", "normal-run", 0.1, 10, 1, "2026-07-17T00:00:00")]
    rows_b[0]["pdf"] = "Shared.pdf"
    rows_b[0]["version"] = "vB"
    cmp = _version_pair_compare(rows_a + rows_b, "vA", "vB", pipeline_runs=[REEVAL_RUNS[0], REEVAL_RUNS[1]])
    cell = cmp["per_pdf"]["shared"]
    assert cell["n_a"] == 1  # gepoolt: 1 Note, nicht 2 Zeilen


def test_build_version_pdf_matrix_accepts_pipeline_runs_kwarg():
    rows = [_qrow(f"n{i}", "normal-run", 0.1, 10, 1, "2026-01-01T00:00:00") for i in range(3)]
    for r in rows:
        r["pdf"] = "a.pdf"
        r["version"] = "v1"
    m = build_version_pdf_matrix(rows, pipeline_runs=[])
    assert m["versions"] == ["v1"]


# ── Server: eval_version-Dropdown-Badge zaehlt Notes, nie Zeilen ───────────
# (Badge-Entscheid Till-Go 2026-07-16: DB-Beleg 81 Zeilen / 27 Notes bei
# eval_version 4.3 -- Konsistenz mit JEDER anderen n-Anzeige im Dashboard
# (KPI-Kachel n_notes, Matrix-n, per-PDF-Tabelle), keine zweite "Notes vs.
# Messungen"-Doppelanzeige nur an dieser einen Stelle.)


def _dbrow_repeat(note, run_id, hall, ts, eval_version="4.3"):
    return {
        "run_id": run_id,
        "note_path": note,
        "pipeline_version": "v0.3.144",
        "version": "v0.3.144",
        "hallucination_rate": hall,
        "anchors_total": 400,
        "anchors_hallucinated": round(hall * 400),
        "coverage_factual": 0.5,
        "pdf": "a.pdf",
        "eval_version": eval_version,
        "timestamp": ts,
    }


def test_eval_version_badge_counts_notes_not_rows_with_repeat_sweeps(monkeypatch):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    evals = []
    for i in range(27):
        note = f"n{i}.md"
        evals.append(_dbrow_repeat(note, "reeval-run", 0.035, f"2026-07-15T23:30:{i:02d}"))
        evals.append(_dbrow_repeat(note, "repeat-run-1", 0.025, f"2026-07-16T15:10:{i:02d}"))
        evals.append(_dbrow_repeat(note, "repeat-run-2", 0.04, f"2026-07-16T16:50:{i:02d}"))
    assert len(evals) == 81  # DB-Beleg: 81 Zeilen

    monkeypatch.setattr(_cfg, "AGENT_VERSION", "v0.3.144")
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: REEVAL_RUNS)
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: evals)
    monkeypatch.setattr(_gdb, "query_archived_pipeline_versions", lambda *a, **k: [])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    data = S.build_data()
    versions = {o["version"]: o["n"] for o in data["available_eval_versions"]}
    assert versions["4.3"] == 27  # Notes, NICHT 81 Zeilen
    assert data["kpis"]["n_notes"] == 27
    # gepoolte Kopfzahl: 40/1200 = 3,3 % (Referenzwert), nicht 4,0 % (nur der
    # juengste Sweep, bisheriges Latest-Wins-Verhalten).
    assert data["kpis"]["avg_hall"] == 3.3


# ── Frontend-Anker (kein bash grep/sed auf dem bewussten NUL-Byte) ─────────
# `internal/dashboard/eval_dashboard.html` enthält ein bewusstes NUL-Byte —
# Zugriff ausschließlich über `_build_live_html()`, nie über bash grep/sed
# (Muster: test_dashboard_reeval_series_flag.py).


def test_html_reeval_note_mentions_pool_over_n_measurements():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function renderPairMatrix")
    end = html.index("/* ── Charts", start)
    block = html[start:end]
    # Bestehender Satz (test_dashboard_reeval_series_flag.py) bleibt
    # unveraendert -- der neue Satz wird ANGEHAENGT, nicht ersetzt.
    assert "Werte messen den Eval-Stand, nicht die Erzeugungsversion der Notes" in block
    assert "Pool über N Wiederholungsmessungen je Note" in block


def test_html_matrix_cell_tooltip_shows_pooled_measurements():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function renderPairMatrix")
    end = html.index("/* ── Charts", start)
    block = html[start:end]
    assert "n_pooled" in block
    assert "pooled_n_measurements" in block
    assert "Wiederholungsmessungen" in block


def test_html_scatter_tooltip_shows_repeat_pooled_spread():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function renderScatter")
    block = html[start : start + 3000]  # Chart.js-Config liegt direkt im Funktionskoerper
    assert "p.repeat_pooled" in block
    assert "Spannweite" in block


def test_html_note_drawer_shows_pooled_measurements():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function _renderNoteDrawer")
    end = html.index("const _drawerKeyOf")
    block = html[start:end]
    assert "p.repeat_pooled" in block
    assert "Wiederholungsmessungen" in block
