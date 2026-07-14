"""Tests für #249 (Rest zu #237, Folge-PR zu #246): zwei verbleibende
Dashboard-Ehrlichkeitslücken bei 0-Eval-/Merge-only-Läufen.

(a) ch1 "Akzeptanzrate je PDF" faerbte einen Balken voll gruen/tone-basiert
    (T.accept-Schwelle), obwohl fuer diese PDF-Gruppe 0 Notes evaluiert wurden
    (n_notes==0, Akzeptanz stammt rein aus Routing-Logs). Das "(n=X)"-Label
    wurde bei n_notes==0 sogar STILLSCHWEIGEND WEGGELASSEN (`nMap[p.l]?...`
    ist bei 0 falsy) -- der einzige ehrliche Hinweis verschwand genau dann,
    wenn er am wichtigsten waere.
    Anker: internal/dashboard/eval_dashboard.html renderCharts()/ch1 (JS,
    Live-Renderpfad -- NICHT das deprecated generative/eval_dashboard.py
    _build_html()).

(b) Gemergte, nicht evaluierte Vault-Notes (Routing "Merge") tauchten im
    Scatter-Chart (ch2, Fehlerquote vs. Belegrate) schlicht nicht auf --
    sie haben kein hall/cov, also keine Koordinate. Fix: n_merge als reales,
    aus den Routing-Logs stammendes Feld pro PDF-Zeile (_calc_pdf_table /
    _accept_from_runs), im Client als eigene, explizit beschriftete
    Legenden-Kategorie neben dem Scatter sichtbar gemacht -- statt sie
    stillschweigend wegzulassen oder implizit als Erfolg zu werten.

Python-Tests decken die Datenschicht ab (SSoT, praezise); die HTML/JS-Marker-
Tests folgen dem in #246 etablierten Muster (String-Assertions auf
`_build_live_html()`).
"""

from __future__ import annotations

from generative.eval_dashboard import _accept_from_runs, _build_log_data, _calc_pdf_table
from generative.eval_dashboard_server import _build_live_html

_CUR = "v0.3.140"


def _q(pdf, note, ver, hall, cov):
    return {
        "pdf": pdf,
        "note_path": note,
        "pipeline_version": ver,
        "hallucination_rate": hall,
        "coverage_factual": cov,
    }


def _run(key, label, ver, n_total, n_vault, n_merge=0, words=None):
    return {
        "key": key,
        "label": label,
        "ver": ver,
        "n_total": n_total,
        "n_vault": n_vault,
        "n_merge": n_merge,
        "accept_pct": round(n_vault / n_total * 100, 1) if n_total else 0.0,
        "words": words,
    }


# ── (b) Datenschicht: n_merge pro PDF-Zeile ─────────────────────────────────


def test_accept_from_runs_returns_merge_count():
    """_accept_from_runs (SSoT fuer accept/accept_n) muss n_merge auf
    DERSELBEN Versions-Basis zurueckgeben -- kein zweiter Aggregations-Pfad."""
    runs = [{"ver": "v0.3.140", "n_total": 3, "n_vault": 1, "n_merge": 2}]
    accept, ver, n, n_merge = _accept_from_runs(runs, None, None)
    assert n_merge == 2


def test_pdf_table_eval_row_exposes_n_merge():
    """Eval-Gruppen-Zeile (hat Eval-Daten) traegt zusaetzlich n_merge aus den
    Routing-Runs derselben Quelle/Version."""
    quality_rows = [_q("Kling - 2020 - Y.pdf", "k1", "v0.3.140", 0.04, 0.90)]
    all_log_runs = [_run("kling", "Kling", "v0.3.140", n_total=3, n_vault=1, n_merge=2)]
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    row = next(r for r in rows if r["key"].startswith("kling"))
    assert row["n_merge"] == 2
    assert row["n_notes"] == 1


def test_pdf_table_routing_only_merge_stub_exposes_n_merge():
    """Repro #237/#249 (Run 20260713-084724, M3): reiner Merge-Stub-Lauf ohne
    jede Eval-Zeile -- landet in der routing_only-Zeile, muss n_merge tragen,
    damit der Client die Note als 'gemergt, nicht evaluiert' ausweisen kann
    statt sie stillschweigend wegzulassen."""
    all_log_runs = [_run("schlebbe", "Schlebbe & Greifeneder", "v0.3.140", n_total=1, n_vault=0, n_merge=1)]
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, [], current_version=_CUR)
    row = next(r for r in rows if r["key"].startswith("schlebbe"))
    assert row["routing_only"] is True
    assert row["n_notes"] == 0
    assert row["n_merge"] == 1


def test_pdf_table_row_without_merge_defaults_to_zero():
    """Gegenprobe: Laeufe ohne Merge-Stubs (Default-Fall) duerfen n_merge nicht
    fehlen lassen oder None liefern -- 0, nicht Absenz."""
    quality_rows = [_q("Afzal - 2017 - X.pdf", "a1", "v0.3.140", 0.05, 0.80)]
    all_log_runs = [_run("afzal", "Afzal", "v0.3.140", n_total=2, n_vault=2)]
    rows = _calc_pdf_table(_build_log_data(all_log_runs), all_log_runs, quality_rows, current_version=_CUR)
    row = next(r for r in rows if r["key"].startswith("afzal"))
    assert row["n_merge"] == 0


# ── (a) ch1: Balken bei n_notes==0 ehrlich kennzeichnen (Live-HTML/JS) ──────


def test_ch1_bar_color_guarded_by_zero_evaluated_notes():
    """Die tone-basierte Balkenfarbe (T.accept-Schwelle: mint/amber/coral)
    darf nur greifen, wenn tatsaechlich Notes evaluiert wurden -- sonst
    neutrale/graue Farbe (analog #246: n_notes-Guard VOR der tone-Logik).

    Marker ist deckungsgleich mit dem konkreten Guard-Ausdruck (nicht nur
    `nMap[p.l]||0`, das bereits VOR dem Fix an anderer Stelle -- der ax1-
    N-Summe -- vorkommt und daher allein nichts beweisen wuerde)."""
    html = _build_live_html()
    assert "(nMap[p.l]||0)===0" in html, "kein n=0-Guard fuer die ch1-Balkenfarbe gefunden"


def test_ch1_label_shows_n_even_when_zero():
    """Vorher: '(n=X)' wurde bei n_notes===0 wegen Truthy-Check (0 ist falsy)
    STILLSCHWEIGEND WEGGELASSEN -- genau dort, wo der Hinweis am wichtigsten
    waere. Fix: expliziter != null-Check statt Truthy-Check."""
    html = _build_live_html()
    assert "nMap[p.l]!=null" in html or "nMap[p.l] != null" in html


def test_ch1_tooltip_explains_missing_eval_basis():
    html = _build_live_html()
    assert "keine Eval-Basis" in html


# ── (b) Scatter/Legende: Merge-Kategorie sichtbar (Live-HTML/JS) ───────────


def test_scatter_legend_has_merged_not_evaluated_category():
    """Gemergte, nicht evaluierte Notes werden im Scatter-Panel als eigene,
    explizit beschriftete Kategorie sichtbar -- nicht stillschweigend
    weggelassen (Issue-Wortlaut)."""
    html = _build_live_html()
    assert "gemergt, nicht evaluiert" in html.lower()


def test_scatter_legend_merge_count_sourced_from_pdf_table():
    """Der Client liest n_merge aus d.pdf_table (SSoT, kein zweiter
    Datenpfad) -- keine neue serverseitige Top-Level-JSON-Struktur."""
    html = _build_live_html()
    assert "r.n_merge" in html or "p.n_merge" in html
