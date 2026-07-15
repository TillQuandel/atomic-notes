"""Tests für Re-Eval-Dedup in den KPI-/Trend-Aggregationen (Statistik-Review 2026-07-15).

Befund (3 unabhängige Opus-Statistiker, konvergent + adversarial bestätigt):
note_evals enthält mehrere Eval-Zeilen derselben Note innerhalb einer
pipeline_version (Re-Evals + identische Duplikat-Inserts; Produktionsbeleg
v0.3.140 = 52 Zeilen / 40 distinct Notes, 12 Duplikate; v0.3.143 = 8 Duplikate).
Ungefiltert poolt jede KPI-/Trend-Aggregation (_calc_kpis, _calc_pdf_table,
kpi_trend im Server) Anker mehrfach-evaluierter Notes mehrfach — Pseudo-
replikation, ~2pp Bias nach unten auf der Fehlerquote (Produktionskopie:
gepoolte Hall-Rate v0.3.140 7,53 % -> 9,46 % nach Dedup, weil oft
re-evaluierte Notes tendenziell gute Raten haben). Zusätzlich: die Kachel
zeigte n=40 (distinct Notes), die Pooling-Basis war aber 52 Zeilen —
inkonsistent.

Fix: pro (pipeline_version, note) NUR die neueste Eval-Zeile (max timestamp,
Tie-Break eval_id) in jede Aggregation. Note-Identität normalisiert über
`_note_key` (Namespace-Prefix `vault__`/`inbox__`/`merge__` gestrippt) — Notes
können zwischen zwei Re-Evals den Namespace wechseln (Routing-Änderung).
"""

from __future__ import annotations

from generative.eval_dashboard import (
    _build_log_data,
    _calc_kpis,
    _calc_pdf_table,
    _dedup_latest_per_note,
    _note_key,
    _pooled_hall_pct,
)


def _qrow(note, ver, hall, total, hallucinated, ts, eval_id=None, cov=0.5):
    return {
        "note_path": note,
        "version": ver,
        "hallucination_rate": hall,
        "anchors_total": total,
        "anchors_hallucinated": hallucinated,
        "coverage_factual": cov,
        "timestamp": ts,
        "eval_id": eval_id or f"{ts}__{note}",
    }


# ── _note_key: Namespace-Prefix-Drift ───────────────────────────────────────


def test_note_key_strips_vault_prefix():
    assert _note_key({"note_path": "vault__Zettelkasten.md"}, 0) == "Zettelkasten.md"


def test_note_key_strips_inbox_prefix():
    assert _note_key({"note_path": "inbox__Foo.md"}, 0) == "Foo.md"


def test_note_key_unifies_prefix_drift_between_reevals():
    # Dieselbe Note kann zwischen zwei Re-Evals den Namespace wechseln
    # (Routing-Änderung) — ohne Normalisierung zählten vault__X und X als
    # zwei Identitäten.
    a = _note_key({"note_path": "vault__Zettelkasten.md"}, 0)
    b = _note_key({"note_path": "Zettelkasten.md"}, 1)
    assert a == b


def test_note_key_falls_back_to_row_index_without_identifier():
    assert _note_key({}, 3) == "__row3"
    assert _note_key({}, 3) != _note_key({}, 4)


# ── _dedup_latest_per_note ───────────────────────────────────────────────────


def test_dedup_keeps_only_latest_row_per_note():
    rows = [
        _qrow("a.md", "v1", 0.0, 17, 0, "2026-06-21T19:50:12"),
        _qrow("a.md", "v1", 0.0, 17, 0, "2026-06-21T21:20:53"),
        _qrow("a.md", "v1", 0.294, 17, 5, "2026-07-05T19:59:18"),  # neueste
    ]
    out = _dedup_latest_per_note(rows)
    assert len(out) == 1
    assert out[0]["hallucination_rate"] == 0.294


def test_dedup_normalizes_namespace_prefix_drift():
    rows = [
        _qrow("vault__X.md", "v1", 0.0, 10, 0, "2026-06-01T00:00:00"),
        _qrow("X.md", "v1", 0.5, 10, 5, "2026-06-02T00:00:00"),  # gleiche Note, Prefix weg
    ]
    out = _dedup_latest_per_note(rows)
    assert len(out) == 1
    assert out[0]["hallucination_rate"] == 0.5


def test_dedup_tie_breaks_deterministically_on_eval_id():
    rows = [
        _qrow("a.md", "v1", 0.0, 10, 0, "2026-06-01T00:00:00", eval_id="20260601-000000__a.md"),
        _qrow("a.md", "v1", 0.5, 10, 5, "2026-06-01T00:00:00", eval_id="20260601-000001__a.md"),
    ]
    out = _dedup_latest_per_note(rows)
    assert len(out) == 1
    assert out[0]["hallucination_rate"] == 0.5  # größerer eval_id gewinnt bei Timestamp-Gleichstand


def test_dedup_rows_without_identifier_all_kept():
    rows = [{"hallucination_rate": 0.0}, {"hallucination_rate": 0.5}]
    out = _dedup_latest_per_note(rows)
    assert len(out) == 2  # synthetische Rows zählen einzeln (Fallback Zeilenindex)


def test_dedup_leaves_distinct_notes_untouched():
    rows = [_qrow("a.md", "v1", 0.0, 10, 0, "t1"), _qrow("b.md", "v1", 0.5, 10, 5, "t2")]
    out = _dedup_latest_per_note(rows)
    assert len(out) == 2


def test_dedup_empty_is_empty():
    assert _dedup_latest_per_note([]) == []


# ── Integration: gepoolte Rate steigt nach Dedup (Produktionsmuster) ────────


def test_pooled_hall_increases_after_dedup_when_duplicates_are_clean():
    # 3 identische "gute" Duplikat-Zeilen + 1 "schlechte" neueste Zeile für
    # dieselbe Note (Produktionsmuster "Asynchronous E-Learning.md",
    # v0.3.140): ungefiltert drückt die 3x wiederholte 0%-Rate die Poolung
    # nach unten.
    rows = [
        _qrow("dup.md", "v1", 0.0, 17, 0, "2026-06-21T19:50:12"),
        _qrow("dup.md", "v1", 0.0, 17, 0, "2026-06-21T21:20:53"),
        _qrow("dup.md", "v1", 0.0, 17, 0, "2026-06-21T21:30:36"),
        _qrow("dup.md", "v1", 0.294, 17, 5, "2026-07-05T19:59:18"),
        _qrow("clean.md", "v1", 0.0, 20, 0, "2026-06-25T00:00:00"),
    ]
    raw_pct = _pooled_hall_pct(rows)  # ungefiltert: 5 / (17*4 + 20) = 5,6 %
    deduped_pct = _pooled_hall_pct(_dedup_latest_per_note(rows))  # dedup: 5 / (17 + 20) = 13,5 %
    assert raw_pct < deduped_pct
    assert deduped_pct == 13.5


# ── _calc_kpis nutzt die deduplizierte Basis ────────────────────────────────


def test_calc_kpis_hall_uses_deduped_basis():
    rows = [
        _qrow("dup.md", "v1", 0.0, 10, 0, "2026-06-21T00:00:00"),
        _qrow("dup.md", "v1", 1.0, 10, 10, "2026-06-22T00:00:00"),  # neueste gewinnt
        _qrow("solo.md", "v1", 0.0, 10, 0, "2026-06-21T00:00:00"),
    ]
    kpis = _calc_kpis({}, [], rows, [], current_version="v1")
    # dedupliziert: dup.md zählt nur mit der neuesten Zeile (10 Anker, 10
    # halluziniert) + solo.md (10 Anker, 0 halluziniert) -> 10/20 = 50 %
    assert kpis["avg_hall"] == 50.0
    assert kpis["n_notes"] == 2
    # Kachel-n muss zur Pooling-Basis passen (Produktionsbefund: Kachel n=40,
    # Pooling-Basis 52 Zeilen — inkonsistent vor dem Fix).
    assert kpis["hall_rows_n"] == 2
    assert kpis["hall_anchors_total"] == 20


def test_calc_kpis_coverage_uses_deduped_basis():
    rows = [
        _qrow("dup.md", "v1", 0.0, 10, 0, "2026-06-21T00:00:00", cov=0.2),
        _qrow("dup.md", "v1", 0.0, 10, 0, "2026-06-22T00:00:00", cov=0.9),  # neueste gewinnt
    ]
    kpis = _calc_kpis({}, [], rows, [], current_version="v1")
    assert kpis["avg_cov"] == 90.0  # nicht Median(20, 90) = 55.0 — nur die neueste Zeile zählt


# ── _calc_pdf_table nutzt dieselbe Basis (SSoT mit der KPI-Kachel) ─────────


def test_calc_pdf_table_hall_uses_deduped_basis():
    quality_rows = [
        {
            "pdf": "Quelle - 2020 - X.pdf",
            "note_path": "dup.md",
            "pipeline_version": "v1",
            "hallucination_rate": 0.0,
            "anchors_total": 10,
            "anchors_hallucinated": 0,
            "coverage_factual": 0.5,
            "timestamp": "2026-06-21T00:00:00",
        },
        {
            "pdf": "Quelle - 2020 - X.pdf",
            "note_path": "dup.md",
            "pipeline_version": "v1",
            "hallucination_rate": 1.0,
            "anchors_total": 10,
            "anchors_hallucinated": 10,
            "coverage_factual": 0.5,
            "timestamp": "2026-06-22T00:00:00",
        },
    ]
    all_log_runs = [
        {
            "key": "quelle-2020",
            "label": "Quelle",
            "ver": "v1",
            "n_total": 1,
            "n_vault": 1,
            "accept_pct": 100.0,
            "words": None,
        }
    ]
    log_data = _build_log_data(all_log_runs)
    rows = _calc_pdf_table(log_data, all_log_runs, quality_rows, current_version="v1")
    row = next(r for r in rows if r["key"].startswith("quelle"))
    assert row["hall"] == 100.0  # nur neueste Zeile (10/10), nicht gepoolt über beide (5/20=25%)
    assert row["n_notes"] == 1


# ── Scatter nutzt dieselbe Basis (Nachbesserung adversariale Kontrolle #293) ─
# Befund: _chart_scatter_versioned deduplizierte NICHT — der Scatter zeigte 52
# Punkte (v0.3.140), während die KPI-Kachel nach Fix 1 korrekt "40 evaluierte
# Notes" sagte; re-evaluierte Notes erschienen doppelt (z. B. "Asynchronous
# E-Learning" bei x=0,0 UND x=29,4). Dieselbe "Instanzen vs. distinct"-
# Bugklasse (#194), die dieser PR schließt.


def test_scatter_shows_distinct_notes_per_version():
    from generative.eval_dashboard_server import _chart_scatter_versioned

    rows = [
        # dup.md: 3 Re-Evals in v1 -> nur der neueste Punkt (x=29.4) darf erscheinen
        _qrow("dup.md", "v1", 0.0, 17, 0, "2026-06-21T19:50:12"),
        _qrow("dup.md", "v1", 0.0, 17, 0, "2026-06-21T21:20:53"),
        _qrow("dup.md", "v1", 0.294, 17, 5, "2026-07-05T19:59:18"),
        _qrow("solo.md", "v1", 0.1, 10, 1, "2026-06-25T00:00:00"),
    ]
    for r in rows:
        r["pdf"] = "Quelle - 2020 - X.pdf"
    out = _chart_scatter_versioned(rows)
    assert len(out["points"]) == 2, f"Scatter zeigt Eval-Instanzen statt distinct Notes: {len(out['points'])} Punkte"
    dup_points = [p for p in out["points"] if p["label"] == "dup"]
    assert len(dup_points) == 1
    assert dup_points[0]["x"] == 29.4  # neueste Eval-Zeile gewinnt, nicht die alte 0,0


def test_scatter_dedup_is_per_version_not_global():
    from generative.eval_dashboard_server import _chart_scatter_versioned

    # Dieselbe Note in ZWEI Versionen bleibt zwei Punkte (der Versions-Filter
    # des Scatters vergleicht Versionen) — Dedup nur INNERHALB einer Version.
    rows = [
        _qrow("a.md", "v1", 0.1, 10, 1, "2026-06-01T00:00:00"),
        _qrow("a.md", "v2", 0.2, 10, 2, "2026-06-02T00:00:00"),
    ]
    for r in rows:
        r["pdf"] = "Quelle - 2020 - X.pdf"
    out = _chart_scatter_versioned(rows)
    assert len(out["points"]) == 2
    assert sorted(p["version"] for p in out["points"]) == ["v1", "v2"]
