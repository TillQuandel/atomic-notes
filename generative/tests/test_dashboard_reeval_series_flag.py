"""Nachbesserung Punkt 4 (Statistiker-Empfehlung, Reviews 2026-07-15/16 zu PR #312):
Re-Eval-Kennzeichnung.

Befund: unter eval_version 4.3 haengen alle 27 note_evals-Zeilen an
`pipeline_version` v0.3.144 -- das ist der CODE-STAND des Re-Eval-Sweeps
(`reeval_baseline.py`, ausgefuehrt zum Zeitpunkt v0.3.144), NICHT die
Erzeugungsversion der Notes. Der zugehoerige Lauf in `pipeline_runs` traegt
dafuer den expliziten Marker `pipeline_version="reeval"` (verifiziert in
generative/reeval_baseline.py: `INSERT OR IGNORE INTO pipeline_runs (run_id,
timestamp, pipeline_version, ...) VALUES (?, datetime('now'), 'reeval', ...)`),
verknuepft ueber `run_id`. Ohne Kennzeichnung liest die Matrix/das Dashboard
"v0.3.144" faelschlich als Erzeugungsversion.

Fix: `D._is_reeval_series(quality_rows, pipeline_runs)` prueft, ob JEDE Zeile
der aktiven eval_version (`_matrix_base_rows`, vor Einzelwert-Filtern) an
einen `pipeline_runs`-Eintrag mit `pipeline_version=="reeval"` haengt. Server
exponiert das Ergebnis als `is_reeval_series` im Payload; das Matrix-Insight
zeigt bei True eine erklaerende Zeile.

`internal/dashboard/eval_dashboard.html` enthält ein bewusstes NUL-Byte --
Zugriff ausschließlich über `_build_live_html()` (Muster:
test_dashboard_version_pdf_matrix.py), nie über bash grep/sed.
"""

from __future__ import annotations

from generative.eval_dashboard import _is_reeval_series

# ── _is_reeval_series (reine Aggregationsfunktion) ─────────────────────────


def test_is_reeval_series_true_when_every_row_maps_to_reeval_marker():
    rows = [{"run_id": "r1"}, {"run_id": "r2"}]
    pipeline_runs = [
        {"run_id": "r1", "pipeline_version": "reeval"},
        {"run_id": "r2", "pipeline_version": "reeval"},
    ]
    assert _is_reeval_series(rows, pipeline_runs) is True


def test_is_reeval_series_false_when_any_row_is_normal_generation():
    """Gemischter Fall: EINE Zeile stammt aus einem normalen Generierungslauf
    (pipeline_version="v0.3.144") -- die eval_version ist dann keine reine
    Re-Eval-Serie, das Flag darf nicht faelschlich True zeigen."""
    rows = [{"run_id": "r1"}, {"run_id": "r2"}]
    pipeline_runs = [
        {"run_id": "r1", "pipeline_version": "reeval"},
        {"run_id": "r2", "pipeline_version": "v0.3.144"},
    ]
    assert _is_reeval_series(rows, pipeline_runs) is False


def test_is_reeval_series_false_when_quality_rows_empty():
    """Leere eval_version -> nichts zu kennzeichnen, kein falsches Positiv."""
    assert _is_reeval_series([], []) is False
    assert _is_reeval_series([], [{"run_id": "r1", "pipeline_version": "reeval"}]) is False


def test_is_reeval_series_false_when_run_id_missing_from_pipeline_runs():
    """Zeile ohne matchenden pipeline_runs-Eintrag (fehlender/unbekannter
    run_id) zaehlt NICHT als reeval -- nur eine explizite Bestaetigung
    (jede Zeile matcht 'reeval') schaltet das Flag."""
    rows = [{"run_id": "unknown"}]
    assert _is_reeval_series(rows, []) is False


def test_is_reeval_series_false_when_row_has_no_run_id():
    rows = [{"note_path": "x.md"}]
    pipeline_runs = [{"run_id": "r1", "pipeline_version": "reeval"}]
    assert _is_reeval_series(rows, pipeline_runs) is False


def test_is_reeval_series_true_when_every_row_maps_to_reeval_repeat_marker():
    """--repeat-Wiederholungs-Sweeps (reeval_baseline.py) tragen den Marker
    'reeval-repeat' statt 'reeval' (eigenes Run-Label, damit die Rauschanalyse
    Wiederholungsgruppen unterscheiden kann). Dieselbe Diagnose (Zeilen
    messen den Eval-Code-Stand des Sweeps, nicht die Notes-Erzeugungsversion)
    gilt fuer Repeat-Sweeps gleichermassen -- das Flag darf nicht stumm
    ausbleiben, nur weil der Marker nicht exakt 'reeval' lautet."""
    rows = [{"run_id": "r1"}, {"run_id": "r2"}]
    pipeline_runs = [
        {"run_id": "r1", "pipeline_version": "reeval-repeat"},
        {"run_id": "r2", "pipeline_version": "reeval-repeat"},
    ]
    assert _is_reeval_series(rows, pipeline_runs) is True


def test_is_reeval_series_true_when_mixed_reeval_and_reeval_repeat_markers():
    """Ein normaler Re-Eval-Sweep und ein Repeat-Sweep zusammen sind beide
    'reeval-like' -- kein normaler Generierungslauf mischt sich rein."""
    rows = [{"run_id": "r1"}, {"run_id": "r2"}]
    pipeline_runs = [
        {"run_id": "r1", "pipeline_version": "reeval"},
        {"run_id": "r2", "pipeline_version": "reeval-repeat"},
    ]
    assert _is_reeval_series(rows, pipeline_runs) is True


# ── Server-Integration: build_data() liefert is_reeval_series ─────────────


def _dbrow(note, ver, pdf, hall, ts, run_id, eval_version="4.3"):
    return {
        "run_id": run_id,
        "note_path": note,
        "pipeline_version": ver,
        "version": ver,
        "hallucination_rate": hall,
        "anchors_total": 10,
        "anchors_hallucinated": 0,
        "coverage_factual": 0.5,
        "pdf": pdf,
        "eval_version": eval_version,
        "timestamp": ts,
    }


def _patched_build_data(monkeypatch, evals, pipeline_runs, **kwargs):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D

    monkeypatch.setattr(_cfg, "AGENT_VERSION", "v0.3.144")
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: pipeline_runs)
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: evals)
    monkeypatch.setattr(_gdb, "query_archived_pipeline_versions", lambda *a, **k: [])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    from generative import eval_dashboard_server as S

    return S.build_data(**kwargs)


def test_build_data_flags_pure_reeval_eval_version(monkeypatch):
    """Nachbau des #232/Punkt-4-Produktionsmusters: eval_version 4.3, alle
    Zeilen haengen an einen einzigen Reeval-Run."""
    evals = [
        _dbrow(f"n{i}", "v0.3.144", "a.pdf", 0.1, f"2026-07-15T00:00:{i:02d}", run_id="reeval-run-1") for i in range(3)
    ]
    pipeline_runs = [
        {"run_id": "reeval-run-1", "pipeline_version": "reeval", "pdf_source": "baseline-reeval"},
    ]
    data = _patched_build_data(monkeypatch, evals, pipeline_runs)
    assert data["is_reeval_series"] is True


def test_build_data_flags_pure_repeat_sweep_eval_version(monkeypatch):
    """--repeat-Variante von test_build_data_flags_pure_reeval_eval_version:
    alle Zeilen haengen an einem Repeat-Sweep-Run ('reeval-repeat')."""
    evals = [
        _dbrow(f"n{i}", "v0.3.144", "a.pdf", 0.1, f"2026-07-16T00:00:{i:02d}", run_id="repeat-run-1") for i in range(3)
    ]
    pipeline_runs = [
        {"run_id": "repeat-run-1", "pipeline_version": "reeval-repeat", "pdf_source": "baseline-reeval"},
    ]
    data = _patched_build_data(monkeypatch, evals, pipeline_runs)
    assert data["is_reeval_series"] is True


def test_build_data_does_not_flag_mixed_eval_version(monkeypatch):
    """Normale eval_version (echte Generierungs-Runs) darf NICHT als
    Re-Eval-Serie markiert werden -- Regressionswaechter gegen ein zu
    aggressives Flag."""
    evals = [
        _dbrow(f"n{i}", "v0.3.144", "a.pdf", 0.1, f"2026-07-15T00:00:{i:02d}", run_id="normal-run-1") for i in range(3)
    ]
    pipeline_runs = [
        {"run_id": "normal-run-1", "pipeline_version": "v0.3.144", "pdf_source": "a.pdf"},
    ]
    data = _patched_build_data(monkeypatch, evals, pipeline_runs)
    assert data["is_reeval_series"] is False


def test_build_data_reeval_flag_scoped_to_active_eval_version(monkeypatch):
    """Das Flag bezieht sich NUR auf die aktive eval_version (_matrix_base_
    rows) -- eine andere eval_version mit normalen Runs darf das Flag der
    aktiven Re-Eval-Serie nicht verwaessern."""
    evals = [
        _dbrow("n1", "v0.3.144", "a.pdf", 0.1, "2026-07-15T00:00:00", run_id="reeval-run-1", eval_version="4.3"),
        _dbrow("m1", "v0.3.140", "a.pdf", 0.1, "2026-06-01T00:00:00", run_id="normal-run-1", eval_version="4.1"),
    ]
    pipeline_runs = [
        {"run_id": "reeval-run-1", "pipeline_version": "reeval"},
        {"run_id": "normal-run-1", "pipeline_version": "v0.3.140"},
    ]
    data = _patched_build_data(monkeypatch, evals, pipeline_runs, eval_version="4.3")
    assert data["eval_version"] == "4.3"
    assert data["is_reeval_series"] is True


# ── Frontend-Anker: Matrix-Insight zeigt den Re-Eval-Hinweis ───────────────


def _pairmatrix_js_block() -> str:
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function renderPairMatrix")
    end = html.index("/* ── Charts", start)
    return html[start:end]


def test_html_has_reeval_note_element():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    assert 'id="pm-reeval-note"' in html


def test_html_render_pair_matrix_shows_reeval_sentence_when_flagged():
    block = _pairmatrix_js_block()
    assert "is_reeval_series" in block
    assert "Re-Eval-Serie" in block
    assert "Werte messen den Eval-Stand, nicht die Erzeugungsversion der Notes" in block
