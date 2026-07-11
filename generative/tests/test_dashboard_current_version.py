"""Tests für die config-verankerte „aktuelle Version"-Wahl (#191).

Befund: KPI-Kacheln (`_calc_kpis`) und Agent-Stats (`_read_agent_stats`) wählten
die „aktuelle" Pipeline-Version per höchster Versionsnummer. Verwaiste
WIP-Branch-Zeilen im geteilten `.cache` (v0.3.141/142 — nie auf master) kaperten
dadurch die Anzeige: Das Dashboard zeigte 10 Eval-Zeilen vom 28.06. als
aktuellen Stand, obwohl der zeitlich letzte Lauf (05.07.) auf v0.3.140 lief.

Fix-Semantik: Anker an `config.AGENT_VERSION`, wenn dafür Daten existieren;
sonst Fallback auf die zeitlich jüngste generative Version mit Daten. Reine
Zeitsortierung reicht nicht — Re-Evals alter Notes schreiben alte Version mit
frischem Timestamp.
"""

from __future__ import annotations

from generative.config import AGENT_VERSION
from generative.eval_dashboard import _calc_kpis, _current_version
from generative.eval_dashboard_server import _current_db_version


def _qrow(ver: str, ts: str, hall: float, total: int, hallucinated: int) -> dict:
    return {
        "version": ver,
        "timestamp": ts,
        "hallucination_rate": hall,
        "anchors_total": total,
        "anchors_hallucinated": hallucinated,
        "coverage_factual": 0.75,
    }


# WIP-Szenario: verwaiste höhere Versionsnummer mit älterem Timestamp
_WIP_ROW = _qrow("v0.3.142", "2026-06-28T10:00:00", 0.2, 10, 2)
_CUR_ROWS = [
    _qrow("v0.3.140", "2026-07-05T20:00:00", 0.0, 10, 0),
    _qrow("v0.3.140", "2026-07-05T20:05:00", 0.1, 10, 1),
]


# ── _current_version (quality_history-Zeilen) ───────────────────────────────


def test_current_version_prefers_config_version_with_data():
    # Höhere WIP-Nummer vorhanden — config-Version hat Daten und gewinnt.
    assert _current_version([_WIP_ROW, *_CUR_ROWS], current="v0.3.140") == "v0.3.140"


def test_current_version_falls_back_to_newest_timestamp():
    # Für die config-Version (frisch gebumpt) existieren noch keine Zeilen →
    # zeitlich jüngste Version mit Daten, NICHT die höchste Nummer.
    assert _current_version([_WIP_ROW, *_CUR_ROWS], current="v0.3.143") == "v0.3.140"


def test_current_version_empty_rows_is_none():
    assert _current_version([], current="v0.3.140") is None


def test_current_version_ignores_extractive_in_fallback():
    rows = [
        _qrow("extractive-v0.2.0", "2026-07-06T09:00:00", 0.0, 5, 0),
        *_CUR_ROWS,
    ]
    # extraktive Pipeline ist jünger, gehört aber nicht in den generativen Trend (#36).
    assert _current_version(rows, current="v9.9.9") == "v0.3.140"


def test_current_version_default_anchors_to_config():
    # Default (current=None) löst config.AGENT_VERSION auf — robust gegen Bumps.
    rows = [_qrow(AGENT_VERSION, "2026-07-05T20:00:00", 0.0, 10, 0), _WIP_ROW]
    assert _current_version(rows) == AGENT_VERSION


# ── _calc_kpis-Integration ──────────────────────────────────────────────────


def test_calc_kpis_version_anchored_not_highest_number():
    kpis = _calc_kpis({}, [], [_WIP_ROW, *_CUR_ROWS], [], current_version="v0.3.140")
    assert kpis["kpi_version"] == "v0.3.140"
    # gepoolt nur über die .140-Zeilen: 1/20 = 5.0 % — die WIP-Zeile (2/10) bleibt draußen
    assert kpis["avg_hall"] == 5.0
    assert kpis["n_notes"] == 2


# ── _current_db_version (pipeline_runs-DB, Agent-Stats) ─────────────────────


def _run(ver: str, ts: str) -> dict:
    return {"run_id": f"r-{ver}-{ts}", "pipeline_version": ver, "timestamp": ts}


def test_db_version_prefers_config_version_with_runs():
    runs = [_run("v0.3.142", "2026-06-28T09:29:13"), _run("v0.3.140", "2026-07-05T19:59:00")]
    assert _current_db_version(runs, current="v0.3.140") == "v0.3.140"


def test_db_version_falls_back_to_newest_run():
    runs = [_run("v0.3.142", "2026-06-28T09:29:13"), _run("v0.3.140", "2026-07-05T19:59:00")]
    assert _current_db_version(runs, current="v0.3.143") == "v0.3.140"


def test_db_version_ignores_extractive():
    runs = [_run("extractive-v0.2.0", "2026-07-06T09:00:00"), _run("v0.3.140", "2026-07-05T19:59:00")]
    assert _current_db_version(runs, current="v9.9.9") == "v0.3.140"


def test_db_version_empty_is_none():
    assert _current_db_version([], current="v0.3.140") is None
