"""Tests für #196 P1–P3: Waisen-Versions-Banner, Kosten pro akzeptierter Note,
prompt-basierte Cache-Effizienz.

P1: Eval-Zeilen/Runs mit Pipeline-Version numerisch ÜBER config.AGENT_VERSION
sind die #191-Fehlerklasse (verwaiste WIP-Branch-Läufe) — das Dashboard meldet
sie als Hinweis-Banner selbst. Dazu: JSONL-Fallback-Flag (deckt nur einen Teil
der DB-Historie ab — Silent-Staleness sichtbar machen).

P2: Kosten pro akzeptierter Note je Version — nur für API-Runs mit Pricing
(cost>0) aussagekräftig; subscription-Läufe (cost 0) und n_vault=0 → None.

P3: cache_pct maß bisher cache_r/(input+output+cache_r) — Output gehört nicht
in den Nenner, cache_creation fehlte (Planner: 75,8 % angezeigt vs. 51,1 %
prompt-basiert). Neu: cache_r/(input+cache_r+cache_c); cache_c wird ausgegeben.
"""

from __future__ import annotations

from generative.eval_dashboard import orphan_versions
from generative.eval_dashboard_server import _cache_pct_prompt_based


# ── P1: orphan_versions ─────────────────────────────────────────────────────


def test_orphan_flags_versions_above_current():
    vers = {"v0.3.140", "v0.3.141", "v0.4.0", "v0.3.99"}
    assert orphan_versions(vers, "v0.3.140") == ["v0.4.0", "v0.3.141"]


def test_orphan_empty_when_all_at_or_below_current():
    assert orphan_versions({"v0.3.140", "v0.3.99", "v0.1.0"}, "v0.3.140") == []


def test_orphan_ignores_foss_and_empty():
    # foss-/extractive-Versionen haben eine eigene Nummernwelt — kein Vergleich.
    vers = {"foss-v9.9.9", "extractive-v2.0", "", None, "v0.3.141"}
    assert orphan_versions(vers, "v0.3.140") == ["v0.3.141"]


def test_orphan_without_current_returns_empty():
    assert orphan_versions({"v0.3.141"}, None) == []
    assert orphan_versions({"v0.3.141"}, "") == []


# ── P3: prompt-basierte Cache-Formel ────────────────────────────────────────


def test_cache_pct_prompt_based_excludes_output_includes_creation():
    # Audit-Fund: input=1000, cache_r=2000, cache_c=900, output=5000.
    # Alt (mit Output, ohne Creation): 2000/8000 = 25 %.
    # Prompt-basiert: 2000/(1000+2000+900) = 51,3 %.
    assert _cache_pct_prompt_based(1000, 2000, 900) == 51.3


def test_cache_pct_prompt_based_zero_denominator():
    assert _cache_pct_prompt_based(0, 0, 0) == 0


# ── P1/P2: build_data-Integration ───────────────────────────────────────────


def _pipeline_run(run_id, ver, cost=0.0, n_vault=3):
    return {
        "run_id": run_id,
        "timestamp": "2026-07-01T00:00:00",
        "pipeline_version": ver,
        "pdf_source": "Testquelle - 2020 - Titel.pdf",
        "pdf_key": "testquelle-2020",
        "pdf_label": "Testquelle - 2020 - Titel",
        "model": "test-model-x",
        "cost_usd": cost,
        "n_generated": 4,
        "n_vault": n_vault,
        "n_inbox": 1,
        "n_merge": 0,
        "n_words": 5000,
        "n_dropped": 0,
        "duration_s": 60.0,
    }


def _eval_row(run_id, ver):
    return {
        "run_id": run_id,
        "note_path": f"notes/{run_id}.md",
        "acceptance_status": "vault",
        "hallucination_rate": 0.05,
        "coverage_factual": 0.8,
        "pipeline_version": ver,
        "version": ver,
        "pdf": "Testquelle - 2020 - Titel.pdf",
        "language": "DE→DE",
        "eval_version": "4.1",
        "anchors_total": 10,
        "anchors_hallucinated": 1,
    }


def _patched_build_data(monkeypatch, runs, evals, current="v0.3.140"):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    monkeypatch.setattr(_cfg, "AGENT_VERSION", current)
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [dict(r) for r in runs])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: [dict(r) for r in evals])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])
    return S.build_data()


def test_build_data_reports_orphan_versions(monkeypatch):
    runs = [
        _pipeline_run("r1", "v0.3.140"),
        _pipeline_run("r2", "v0.3.152"),  # Waise: über AGENT_VERSION
    ]
    evals = [_eval_row("r1", "v0.3.140"), _eval_row("r2", "v0.3.152")]
    data = _patched_build_data(monkeypatch, runs, evals, current="v0.3.140")
    assert data["warnings"]["orphan_versions"] == ["v0.3.152"]
    assert data["warnings"]["current_version"] == "v0.3.140"
    assert data["warnings"]["jsonl_fallback"] is False


def test_build_data_no_orphans_is_empty_list(monkeypatch):
    runs = [_pipeline_run("r1", "v0.3.140")]
    evals = [_eval_row("r1", "v0.3.140")]
    data = _patched_build_data(monkeypatch, runs, evals, current="v0.3.140")
    assert data["warnings"]["orphan_versions"] == []


def test_build_data_flags_jsonl_fallback(monkeypatch):
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    def _boom(*a, **k):
        raise RuntimeError("DB weg")

    monkeypatch.setattr(_gdb, "query_note_evals", _boom)
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [])
    monkeypatch.setattr(D, "_read_quality_history", lambda: [_eval_row("r1", "v0.3.140")])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])
    data = S.build_data()
    assert data["warnings"]["jsonl_fallback"] is True


def test_build_data_cost_per_accepted_note(monkeypatch):
    # v0.3.139: API-Lauf mit Kosten 1.2 $ und 3 Vault-Notes → 0.4 $/Note.
    # v0.3.140: subscription (cost 0) → None („–", kein falsches 0 $).
    runs = [
        _pipeline_run("r1", "v0.3.139", cost=1.2, n_vault=3),
        _pipeline_run("r2", "v0.3.140", cost=0.0, n_vault=3),
    ]
    evals = [_eval_row("r1", "v0.3.139"), _eval_row("r2", "v0.3.140")]
    data = _patched_build_data(monkeypatch, runs, evals)
    cpn = dict(zip(data["kpi_trend"]["versions"], data["kpi_trend"]["cost_per_note"]))
    assert cpn["v0.3.139"] == 0.4
    assert cpn["v0.3.140"] is None


def test_build_data_cost_per_note_none_without_vault_notes(monkeypatch):
    runs = [_pipeline_run("r1", "v0.3.140", cost=2.0, n_vault=0)]
    evals = [_eval_row("r1", "v0.3.140")]
    data = _patched_build_data(monkeypatch, runs, evals)
    cpn = dict(zip(data["kpi_trend"]["versions"], data["kpi_trend"]["cost_per_note"]))
    assert cpn["v0.3.140"] is None
