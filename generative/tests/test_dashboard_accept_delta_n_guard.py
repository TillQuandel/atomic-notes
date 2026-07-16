"""Punkt 4 (D3, Reviews 15.07.): accept-Delta-Guard nutzt falsches n-Array.

Befund: `version_delta()` wird fuer ALLE KPI-Metriken (hall/cov/n/accept/dur/
tokens/cost) mit demselben `kpi_trend["n"]`-Array als Reliability-Guard
aufgerufen (kpi_trend["deltas"] = {m: D.version_delta(kpi_trend, m) for m in
(...)}). `kpi_trend["n"]` ist die Zahl EVALUIERTER Notes (distinct, per LLM-
Eval) -- fuer hall/cov ist das der richtige Nenner (das SIND Eval-Metriken).
Fuer "accept" (Akzeptanzrate = generierte -> in den Vault uebernommene
Notes, `_pooled_accept`, gepoolt aus ALLEN Log-Runs/Routing-Entscheidungen,
nicht nur den LLM-evaluierten) ist der Nenner falsch: die Zahl GEROUTETER
Notes (n_total aus den Pipeline-Runs) ist typischerweise um ein Vielfaches
groesser als die Zahl der LLM-evaluierten Notes (nur eine Stichprobe wird
evaluiert). Der Bug ist "aktuell konservativ" (zeigt reliable:false, obwohl
das Accept-Delta auf hunderten gerouteten Notes beruht) -- keine falschen
Deltas, aber unnoetig graue/unbelastbare Chips.

Fix: `version_delta()` bekommt einen optionalen `n_field`-Parameter (Default
"n" -- rueckwaertskompatibel fuer hall/cov/dur/tokens/cost); der Server
uebergibt fuer "accept" `n_field="accept_n"` (neues kpi_trend-Feld: Summe der
n_total-Werte je Version aus denselben accept_pairs_by_ver, die auch
_pooled_accept speist -- SSoT, keine zweite Zaehlung)."""

from __future__ import annotations

from generative.eval_dashboard import version_delta


def _kpi_trend(**over):
    base = {
        "versions": ["v1", "v2"],
        "accept": [70.0, 80.0],
        "n": [8, 9],  # nur 8/9 LLM-evaluierte Notes -- unter _DELTA_MIN_N=20
        "accept_n": [150, 160],  # aber 150/160 GEROUTETE Notes -- weit ueber 20
    }
    base.update(over)
    return base


def test_accept_delta_uses_n_field_override_not_evaluated_n():
    """Kern-Regression: mit dem falschen ('n') Array waere reliable=False
    (8/9 < 20). Mit dem richtigen ('accept_n') Array ist es reliable=True."""
    d = version_delta(_kpi_trend(), "accept", n_field="accept_n")
    assert d["reliable"] is True
    assert d["reason"] is None


def test_accept_delta_without_override_falls_back_to_n_and_is_conservative():
    """Ohne den Fix (Default n_field='n') bleibt das Delta unreliable --
    Beleg fuer den beschriebenen Bug/die Konservativitaet, kein Verhaltens-
    wechsel am Default (Rueckwaertskompatibilitaet fuer hall/cov/etc.)."""
    d = version_delta(_kpi_trend(), "accept")
    assert d["reliable"] is False
    assert d["reason"] == "n_lt_20"


def test_hall_delta_default_n_field_unchanged():
    """Regressions-Wächter: der Default fuer alle anderen Metriken (hall/cov/
    dur/tokens/cost) bleibt exakt das bisherige Verhalten -- kein n_field noetig."""
    trend = {"versions": ["v1", "v2"], "hall": [9.0, 8.0], "n": [25, 30]}
    d = version_delta(trend, "hall")
    assert d["reliable"] is True


def test_n_field_missing_key_behaves_like_missing_n_array():
    """Falls `accept_n` (aelterer Aufrufer/Test) fehlt: leeres Array, kein
    Crash -- Guard verhaelt sich wie n=0 (nicht reliable), analog zum
    bestehenden pdf_notes-Rueckwaertskompatibilitaets-Verhalten."""
    trend = {"versions": ["v1", "v2"], "accept": [70.0, 80.0]}
    d = version_delta(trend, "accept", n_field="accept_n")
    assert d["reliable"] is False


# ── Server-Integration: build_data() liefert kpi_trend["accept_n"] ─────────


def _eval(note, ver, pdf, hall, ts, eval_version="4.1"):
    return {
        "run_id": f"r-{note}",
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


def _log_run(ver, n_total, n_vault, key="a"):
    return {
        "key": key,
        "label": key,
        "ver": ver,
        "n_total": n_total,
        "n_vault": n_vault,
        "n_merge": 0,
        "n_inbox": n_total - n_vault,
        "accept_pct": round(n_vault / n_total * 100, 1) if n_total else 0.0,
        "words": 1000,
        "pages": 5,
        "chunks": 3,
    }


def test_build_data_exposes_accept_n_as_routed_notes_sum(monkeypatch):
    """accept_n MUSS aus denselben Log-Runs stammen wie _pooled_accept (SSoT)
    -- hier: v1 hat 2 Runs mit zusammen 150 gerouteten Notes, v2 einen Run
    mit 160."""
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    evals = [_eval(f"n{i}", "v1", "a.pdf", 0.1, f"2026-01-01T00:00:{i:02d}") for i in range(9)]
    evals += [_eval(f"m{i}", "v2", "a.pdf", 0.1, f"2026-02-01T00:00:{i:02d}") for i in range(8)]
    runs = [_log_run("v1", 100, 70), _log_run("v1", 50, 35), _log_run("v2", 160, 130)]

    monkeypatch.setattr(_cfg, "AGENT_VERSION", "v2")
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: evals)
    monkeypatch.setattr(_gdb, "query_archived_pipeline_versions", lambda *a, **k: [])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: runs)
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    data = S.build_data()
    kt = data["kpi_trend"]
    assert kt["versions"] == ["v1", "v2"]
    assert kt["n"] == [9, 8]  # evaluierte Notes -- unter 20, "konservativ"
    assert kt["accept_n"] == [150, 160]  # gerouteted Notes -- SSoT mit _pooled_accept
    # Kern-Nachweis: das accept-Delta ist trotz n<20 (evaluiert) reliable,
    # weil es auf accept_n (150/160 geroutet) gehartet ist.
    vd = kt["deltas"]["accept"]
    assert vd["reliable"] is True
    assert vd["reason"] is None
