"""Tests für die Dashboard-Quick-Win-Fixes (Review 2026-06-10).

Befunde aus dem Eval-Dashboard-Review:
- F1: `_median` (Upper-Median ohne Interpolation) widerspricht `statistics.median`
  im kpi_trend — dieselbe Metrik zeigt zwei Werte im UI (KPI 11.1 vs. Spark 9.7).
- F8: `_calc_kpis` liefert kein `total_dropped` → Strip-Zelle "Dropped" immer 0.
- F5: `_read_agent_stats` zählt Stage-8-Eval-Judges (eval_quality_*) als
  Pipeline-Agenten; `_live_run_data` mappt sie korrekt auf "eval".
- F2/F9/F15: Kalibrierungs-View joint Human-Labels gegen eine beliebige
  note_evals-Zeile statt `calibration_labels.llm_hall_rate` zu nutzen;
  coverage_factual NULL wird 0.0 statt None; eval_version hartkodiert '4.1'.
"""

from __future__ import annotations

import sqlite3

import pytest

from generative import db
from generative.eval_dashboard import _calc_kpis, _median
from generative.eval_dashboard_server import _read_calibration_data


# ── F1: Median ──────────────────────────────────────────────────────────────


def test_median_interpolates_even_n():
    # Upper-Median lieferte 11.1; statistics.median interpoliert auf 9.7.
    assert _median([8.3, 11.1]) == pytest.approx(9.7)


def test_median_odd_n_unchanged():
    assert _median([1.0, 5.0, 9.0]) == 5.0


# ── F8: total_dropped ───────────────────────────────────────────────────────


def _log_run(**over):
    base = {
        "key": "bates",
        "label": "Bates 2017",
        "ver": "v0.3.135",
        "n_total": 4,
        "n_vault": 2,
        "n_merge": 1,
        "n_inbox": 1,
        "accept_pct": 50.0,
        "words": 9000,
        "pages": 12,
    }
    base.update(over)
    return base


def test_calc_kpis_sums_total_dropped():
    runs = [_log_run(n_dropped=2), _log_run(n_dropped=3)]
    kpis = _calc_kpis({}, runs, [], [])
    assert kpis["total_dropped"] == 5


def test_calc_kpis_total_dropped_zero_when_field_missing():
    kpis = _calc_kpis({}, [_log_run()], [], [])
    assert kpis["total_dropped"] == 0


# ── F5: Agent-Mapping ───────────────────────────────────────────────────────


def test_canonical_agent_maps_eval_judges():
    from generative.eval_dashboard_server import _canonical_agent

    assert _canonical_agent("eval_quality_v3_audit") == "eval"
    assert _canonical_agent("eval_quality_v3_primary") == "eval"


def test_canonical_agent_drops_non_agents():
    from generative.eval_dashboard_server import _canonical_agent

    for name in ("", "unknown", "?", "orchestrator"):
        assert _canonical_agent(name) is None


def test_canonical_agent_keeps_pipeline_agents():
    from generative.eval_dashboard_server import _canonical_agent

    assert _canonical_agent("planner") == "planner"
    assert _canonical_agent("extractor") == "extractor"


# ── F2/F9/F15: Kalibrierung ────────────────────────────────────────────────

NOTE = "vault__Testnote.md"


def _seed_db(path, cov_rate=None):
    db.init_db(path)
    conn = sqlite3.connect(path)

    def eval_row(hall, cov_factual, pver, ts):
        conn.execute(
            "INSERT INTO note_evals (run_id, note_path, hallucination_rate, "
            "coverage_factual, coverage_rate, pipeline_version, pdf, "
            "eval_version, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"run-{ts}", NOTE, hall, cov_factual, cov_rate, pver, "Bates.pdf", "4.1", ts),
        )

    # Zwei Evals derselben Note: das Label gehört zum 0.30-Eval,
    # die 0.0-Zeile kommt ZULETZT (gewann bisher den dict-Join).
    eval_row(0.30, None, "v0.3.130", "2026-06-01")
    eval_row(0.0, None, "v0.3.135", "2026-06-02")
    conn.execute(
        "INSERT INTO calibration_labels (note_path, eval_version, labeled_at, "
        "n_claims, n_supported, n_hallucinated, n_uncertain, human_hall_rate, "
        "llm_hall_rate, agreement_rate) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (NOTE, "4.1", "2026-06-10", 12, 10, 2, 0, 0.167, 0.30, 0.85),
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def calib_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    _seed_db(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def test_calibration_labeled_rows_use_label_llm_rate(calib_db):
    # Gelabelte Note: llm_hall muss aus calibration_labels.llm_hall_rate kommen
    # (die zum Label-Zeitpunkt passende Rate), nicht aus einer beliebigen
    # note_evals-Zeile derselben Note.
    rows = _read_calibration_data()["rows"]
    labeled = [r for r in rows if r["status"] == "labeled"]
    assert len(labeled) == 1
    assert labeled[0]["llm_hall"] == 30.0
    assert labeled[0]["human_hall"] == 16.7


def test_calibration_cov_none_stays_none(calib_db):
    # coverage_factual UND coverage_rate NULL → None, nicht 0.0 (gleiche
    # Bug-Klasse wie avg_agree) — echter "keine Daten"-Fall.
    rows = _read_calibration_data()["rows"]
    assert all(r["llm_cov"] is None for r in rows)


def test_calibration_cov_falls_back_to_coverage_rate(tmp_path, monkeypatch):
    # #233: coverage_factual ist seit v4 immer NULL (abgeschaffte v1.3-Metrik);
    # _read_calibration_data muss auf coverage_rate zurückfallen statt die
    # LLM-Coverage-Spalte durchgehend leer zu lassen.
    path = tmp_path / "test.db"
    _seed_db(path, cov_rate=0.42)
    monkeypatch.setattr(db, "DB_PATH", path)

    rows = _read_calibration_data()["rows"]
    assert all(r["llm_cov"] == pytest.approx(42.0) for r in rows)


def test_calibration_eval_version_parameter(calib_db):
    # eval_version war hartkodiert '4.1' — bei anderer Version keine 4.1-Daten zeigen.
    data = _read_calibration_data(eval_version="1.3")
    assert data["n_eval"] == 0
    assert data["rows"] == []


# ── F11/F12: Mock-Pfad entfernt, Server-HTML lädt nur live ────────────────


def test_build_live_html_has_no_mock_path():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    assert "MOCK_DATA" not in html
    assert '<script src="data.js"' not in html
    # Live-Boot: Daten via fetch + 15s-Refresh
    # Poll ruft loadAndRender(false) — manual=false unterdrueckt die
    # Refresh-Dimmung beim automatischen Poll (Nachbesserung #204/#221 P-a).
    assert "loadAndRender();" in html
    assert "setInterval(() => loadAndRender(false), 15000)" in html


def test_calibration_counts_respect_note_filter(calib_db):
    # Codex-Finding: rows wurden gefiltert, aber n_eval/n_labeled/has_labels
    # zaehlten weiter die Gesamtmenge → Strip zeigte 301 bei 10 Zeilen.
    data = _read_calibration_data(allowed_note_paths=set())
    assert data["rows"] == []
    assert data["n_eval"] == 0
    assert data["n_labeled"] == 0
    assert data["has_labels"] is False


# ── #233 Folge (Frontend, urspr. #255): LLM-Coverage-Spalte im Kalibrierungs-
# View. Datenschicht liefert `llm_cov` bereits (PR #250), aber #calib-table
# rendert dafuer bisher keine Spalte -> Backend-Wert kommt beim Nutzer nie an.


def test_calibration_table_header_has_llm_coverage_column():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    calib_section = html.split('id="calib-table"')[1].split("</table>")[0]
    assert "LLM-Coverage" in calib_section
    # Tooltip muss die deprecatete Quelle + den Fallback erklaeren (#233-Kernbefund).
    assert "coverage_factual" in calib_section
    assert "coverage_rate" in calib_section
    assert "v1.3" in calib_section


def test_calibration_row_render_uses_llm_cov_with_fmtde():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    render_fn = html.split("function renderCalibration(calib)")[1].split("\nlet _calibShowAll")[0]
    assert "r.llm_cov" in render_fn
    assert "_fmtDE(" in render_fn
    # Leere/NULL-Werte defensiv als Strich rendern (eigene Formatfunktion,
    # analog fmtHall/fmtAgree in derselben Funktion).
    assert "fmtCov" in render_fn


def test_calibration_table_colspan_matches_seven_columns():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    # Neue Spalte -> #calib-table hat jetzt 7 statt 6 Spalten. Alle drei
    # Platzhalterzeilen (Loading, "keine Daten", "weitere anzeigen"-Button)
    # muessen mitziehen, sonst verzieht sich die Zeile (kollabiert nicht,
    # aber falsch ausgerichtet).
    assert html.count('colspan="6"') == 0
    assert html.count('colspan="7"') == 3
