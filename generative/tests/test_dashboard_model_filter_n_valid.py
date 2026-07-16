"""Punkt 2 (Reviews 15.07.): n-valid-Badge je Modell-Filter-Option.

Befund: Das Modell-Dropdown zeigt nur Modellnamen, keine Zeilenzahl. Die
Task-Vorgabe: Optionen sollen die Anzahl VALIDER Eval-Zeilen
(hallucination_rate >= 0) je Modell zeigen -- Gemini-Fehllesungs-Schutz:
dort tragen Zeilen den bestehenden -1.0-Sentinel fuer "ungueltig"
(vgl. _chart_scatter/_matrix_cell_stats), n_valid muss fuer so ein Modell
0 zeigen, nicht die volle (aber wertlose) Zeilenzahl.

Modell-Zuordnung: note_evals traegt KEIN eigenes "model"-Feld (das lebt in
pipeline_runs) -- Join ueber run_id -> token_runs (bereits mit .model
angereichert, s. build_data()) -> quality_rows.run_id, derselbe Mechanismus
wie der bestehende Modell-EINZELWERT-Filter (Server-Kommentar: "Alle Filter
auf token_runs anwenden -> run_ids extrahieren -> quality_rows ... ebenfalls
filtern")."""

from __future__ import annotations


def _token_run(run_id, **over):
    base = {
        "date": "01.01 00:00",
        "run_id": run_id,
        "pdf_label": "",
        "tokens_in": 100,
        "tokens_out": 50,
        "tokens_cache": 0,
        "tokens_cache_read": 0,
        "tokens_cache_create": 0,
        "duration_min": 1.0,
        "calls": 1,
    }
    base.update(over)
    return base


def _pipeline_run(run_id, model, ver="v0.3.144", **over):
    base = {
        "run_id": run_id,
        "model": model,
        "pipeline_version": ver,
        "pdf_label": "a.pdf",
        "pdf_source": "a.pdf",
        "cost_usd": 0.01,
        "wall_clock_s": 10.0,
    }
    base.update(over)
    return base


def _eval_row(note, run_id, hall, ts, ver="v0.3.144", eval_version="4.1"):
    return {
        "run_id": run_id,
        "note_path": note,
        "pipeline_version": ver,
        "version": ver,
        "hallucination_rate": hall,
        "anchors_total": 10,
        "anchors_hallucinated": 0,
        "coverage_factual": 0.5,
        "pdf": "a.pdf",
        "eval_version": eval_version,
        "timestamp": ts,
    }


def _patched_build_data(monkeypatch, evals, pipeline_runs, token_runs, current_version="v0.3.144", **kwargs):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    monkeypatch.setattr(_cfg, "AGENT_VERSION", current_version)
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: pipeline_runs)
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: evals)
    monkeypatch.setattr(_gdb, "query_archived_pipeline_versions", lambda *a, **k: [])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: token_runs)
    return S.build_data(**kwargs)


def test_all_models_carries_n_valid_count_excludes_negative_sentinel(monkeypatch):
    """Anthropic-Modell: 3 gueltige + 1 Sentinel-Zeile (-1.0) -> n_valid=3.
    Gemini-Modell (Fehllesungs-Schutz): 2 Zeilen, BEIDE Sentinel -> n_valid=0,
    nicht 2 (die Zeilen existieren, sind aber wertlos)."""
    pipeline_runs = [
        _pipeline_run("r-claude", "anthropic/claude-sonnet-4-6"),
        _pipeline_run("r-gemini", "gemini/gemini-2.5-flash"),
    ]
    token_runs = [_token_run("r-claude"), _token_run("r-gemini")]
    evals = [
        _eval_row("n1", "r-claude", 0.1, "2026-01-01T00:00:01"),
        _eval_row("n2", "r-claude", 0.2, "2026-01-01T00:00:02"),
        _eval_row("n3", "r-claude", 0.0, "2026-01-01T00:00:03"),
        _eval_row("n4", "r-claude", -1.0, "2026-01-01T00:00:04"),  # Sentinel
        _eval_row("n5", "r-gemini", -1.0, "2026-01-01T00:00:05"),
        _eval_row("n6", "r-gemini", -1.0, "2026-01-01T00:00:06"),
    ]
    data = _patched_build_data(monkeypatch, evals, pipeline_runs, token_runs)
    by_model = {o["model"]: o["n_valid"] for o in data["all_models"]}
    assert by_model["anthropic/claude-sonnet-4-6"] == 3
    assert by_model["gemini/gemini-2.5-flash"] == 0


def test_all_models_n_valid_none_hallucination_not_counted_as_valid(monkeypatch):
    pipeline_runs = [_pipeline_run("r1", "anthropic/claude-haiku-4-5")]
    token_runs = [_token_run("r1")]
    evals = [
        _eval_row("n1", "r1", 0.1, "2026-01-01T00:00:01"),
        _eval_row("n2", "r1", None, "2026-01-01T00:00:02"),
    ]
    data = _patched_build_data(monkeypatch, evals, pipeline_runs, token_runs)
    by_model = {o["model"]: o["n_valid"] for o in data["all_models"]}
    assert by_model["anthropic/claude-haiku-4-5"] == 1


def test_all_models_still_excludes_denylisted_smoke_models(monkeypatch):
    """Regressions-Wächter: die bestehende _MODEL_DENYLIST-Filterung
    (Smoke-/Testmodelle) bleibt unveraendert bei der Umstellung auf
    {model, n_valid}-Objekte."""
    pipeline_runs = [_pipeline_run("r1", "smoke-model"), _pipeline_run("r2", "anthropic/claude-opus-4-7")]
    token_runs = [_token_run("r1"), _token_run("r2")]
    evals = [_eval_row("n1", "r2", 0.1, "2026-01-01T00:00:01")]
    data = _patched_build_data(monkeypatch, evals, pipeline_runs, token_runs)
    models = {o["model"] for o in data["all_models"]}
    assert models == {"anthropic/claude-opus-4-7"}


# ── Frontend-Anker: Dropdown zeigt "(n=X)", value bleibt der Modellname ────


def test_html_model_filter_shows_n_valid_badge_in_option_label():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function _initGlobalModelFilter")
    end = html.index("\n}", start)
    block = html[start:end]
    assert "m.model" in block
    assert "m.n_valid" in block
