"""#238: drei widersprüchliche Token-Totals je Lauf — Billable als Leitzahl.

Befund (Issue #238, nachgerechnet an M2 `20260713-082517`):
  1. KPI-Kachel `cur_tokens` „Pipeline-Tokens" = in+out ALLER Calls inkl.
     Eval-Judges, OHNE Cache (138 673) — eval_dashboard.py `_calc_kpis`.
  2. DB `pipeline_runs.tokens_total` = Pipeline-Agenten in+out+cache, OHNE
     Eval-Judges (1 452 215) — orchestrator.py, vor Stage-8 geschrieben.
  3. Echter billable Total (Trace, alle Calls inkl. Cache inkl. Eval,
     1 802 471) — `eval_agent_stats.run_totals` nach Stage-8, bisher nirgends
     als Kachel sichtbar.

Fix-Richtung (Issue): Billable (in+out+cache_r+cache_c, gleiche Datenquelle
wie `cur_tokens` — `token_runs` aus `_read_token_runs()`, das bereits ALLE
Calls inkl. Eval-Judges liest) als eigene Kachel neben der umbenannten
"Pipeline-Tokens"-Kachel. Kein Umbau der Datenerfassung — nur konsistente
Auswahl (Cache war in `token_runs` schon vorhanden, nur nie summiert) +
Labeling.
"""

from __future__ import annotations

import json

from generative.eval_dashboard import _calc_kpis


def _quality_row(ver="v0.3.140"):
    return {"version": ver, "pipeline_version": ver, "timestamp": "2026-07-13T00:00:00"}


def _token_run(run_id, tin, tout, tcache, ver="v0.3.140"):
    return {
        "run_id": run_id,
        "ver": ver,
        "tokens_in": tin,
        "tokens_out": tout,
        "tokens_cache": tcache,
        "duration_min": 1.0,
        "calls": 1,
        "db_matched": True,
    }


# ── _calc_kpis: Billable = in+out+cache, aus derselben token_runs-Quelle ────
# wie cur_tokens (kein neuer Datenpfad).


def test_cur_tokens_billable_includes_cache():
    runs = [_token_run("r1", tin=100, tout=50, tcache=1000)]
    kpis = _calc_kpis({}, [], [_quality_row()], runs, current_version="v0.3.140")
    assert kpis["cur_tokens"] == 150  # unveraendert: in+out, ohne Cache
    assert kpis["cur_tokens_billable"] == 1150  # in+out+cache


def test_cur_tokens_billable_sums_across_runs_of_current_version():
    runs = [
        _token_run("r1", tin=100, tout=50, tcache=500),
        _token_run("r2", tin=200, tout=80, tcache=700),
        _token_run("r-other-ver", tin=999, tout=999, tcache=999, ver="v0.2.0"),
    ]
    kpis = _calc_kpis({}, [], [_quality_row()], runs, current_version="v0.3.140")
    assert kpis["cur_tokens_billable"] == (100 + 50 + 500) + (200 + 80 + 700)


def test_cur_tokens_cache_breakdown_field_present():
    # Aufschluesselung fuer den Kachel-Hint: cur_tokens_cache separat verfuegbar,
    # nicht nur in der billable-Summe versteckt.
    runs = [_token_run("r1", tin=100, tout=50, tcache=1000)]
    kpis = _calc_kpis({}, [], [_quality_row()], runs, current_version="v0.3.140")
    assert kpis["cur_tokens_cache"] == 1000


def test_cur_tokens_billable_zero_when_no_runs():
    kpis = _calc_kpis({}, [], [], [])
    assert kpis["cur_tokens_billable"] == 0


def test_cur_tokens_in_out_cache_breakdown_fields():
    # Issue-Fix-Vorschlag: "Aufschlüsselung in/out/cache_r/cache_c statt einer
    # Summe zeigen" — die einzelnen Komponenten muessen als eigene KPI-Felder
    # verfuegbar sein (fuer den Billable-Kachel-Hint), nicht nur die Totale.
    runs = [
        {
            "run_id": "r1",
            "ver": "v0.3.140",
            "tokens_in": 100,
            "tokens_out": 50,
            "tokens_cache": 1000,
            "tokens_cache_read": 700,
            "tokens_cache_create": 300,
            "duration_min": 1.0,
            "calls": 1,
            "db_matched": True,
        }
    ]
    kpis = _calc_kpis({}, [], [_quality_row()], runs, current_version="v0.3.140")
    assert kpis["cur_tokens_in"] == 100
    assert kpis["cur_tokens_out"] == 50
    assert kpis["cur_tokens_cache_read"] == 700
    assert kpis["cur_tokens_cache_create"] == 300


# ── _read_token_runs: cache_read/cache_create additiv getrennt verfuegbar ───
# (bisher nur kombiniert als tokens_cache) — fuer die Aufschluesselung im
# Kachel-Hint. Rein additiv: tokens_cache bleibt unveraendert bestehen.


def _write_jsonl(runs_dir, name, records):
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / name).write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


# ── HTML-Label-Ehrlichkeit: Billable ist eine eigene, sichtbare Kachel; die ─
# alte "Pipeline-Tokens"-Kachel ist praezise umbenannt+relabelt; ins-cost
# zaehlt Cache mit (statt es stillschweigend auszulassen).


def test_html_adds_billable_tile_and_relabels_pipeline_tokens():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    # Alte, irrefuehrende Bezeichnung darf nicht mehr vorkommen (enthielt
    # Eval-Judges UND liess Cache weg — beides am Namen "Pipeline-Tokens"
    # nicht erkennbar).
    assert "label:'Pipeline-Tokens'" not in html
    assert "label:'Billable Tokens'" in html
    assert "label:'Tokens (ohne Cache)'" in html
    assert "kpis.cur_tokens_billable" in html


def test_html_ins_cost_insight_counts_cache():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    # #238-Befund: ins-cost liess Cache aus (~9x-Untertreibung) — totalTok
    # muss tok.tokens_cache mitzaehlen.
    assert "(tok.tokens_cache||[]).reduce" in html


def test_read_token_runs_exposes_cache_read_and_create_separately(monkeypatch, tmp_path):
    from generative import eval_dashboard as ed

    runs = tmp_path / "runs"
    _write_jsonl(
        runs,
        "20260713-082517.jsonl",
        [
            {
                "model": "sonnet",
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_tokens": 700,
                "cache_creation_tokens": 300,
            },
        ],
    )
    monkeypatch.setattr(ed, "RUNS_DIR", runs)

    result = ed._read_token_runs()

    assert len(result) == 1
    # Bestehendes kombiniertes Feld bleibt unveraendert (kein Breaking Change
    # fuer die bestehenden Chart-Konsumenten, z. B. ch5-Stapel).
    assert result[0]["tokens_cache"] == 1000
    # Neue, additive Aufschluesselung.
    assert result[0]["tokens_cache_read"] == 700
    assert result[0]["tokens_cache_create"] == 300
