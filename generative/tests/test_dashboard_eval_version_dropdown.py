"""Tests für Punkt 1 (Till-Wunsch + Zusatzbefund 2026-07-16): eval_version-Dropdown.

Befund: Das Dashboard defaultete unveränderlich auf die neueste eval_version
(4.3). Der URL-Param `?eval_version=4.1` wirkte im Browser NICHT — der
Client-Fetch von `/data.json` reichte ihn nie durch (Kommentar im Bestand:
"Eval-Version-Filter entfernt: nie aus URL/State setzen"). Fix: Dropdown in
der Filterbar (Optionen aus `available_eval_versions`, inkl. Zeilenzahl je
Version für die Anzeige "4.3 (n=27)"), das den `eval_version`-Param in JEDEN
`/data.json`-Fetch übernimmt (inkl. 15s-Auto-Refresh).

`internal/dashboard/eval_dashboard.html` enthält ein bewusstes NUL-Byte —
Zugriff ausschließlich über `_build_live_html()`/`Path.read_text(encoding=
"utf-8")`, nie über bash grep/sed.
"""

from __future__ import annotations

import pytest


def _eval(note, ver, pdf, hall, ts, eval_version="4.1", total=10, hallucinated=0, cov=0.5):
    return {
        "run_id": f"r-{note}",
        "note_path": note,
        "pipeline_version": ver,
        "version": ver,
        "hallucination_rate": hall,
        "anchors_total": total,
        "anchors_hallucinated": hallucinated,
        "coverage_factual": cov,
        "pdf": pdf,
        "eval_version": eval_version,
        "timestamp": ts,
    }


def _patched_build_data(monkeypatch, evals, current_version="v0.3.144", **kwargs):
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D

    monkeypatch.setattr(_cfg, "AGENT_VERSION", current_version)
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: evals)
    monkeypatch.setattr(_gdb, "query_archived_pipeline_versions", lambda *a, **k: [])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    from generative import eval_dashboard_server as S

    return S.build_data(**kwargs)


# ── Server: available_eval_versions traegt Zeilenzahl je Version ──────────


def test_available_eval_versions_carries_row_count_per_version(monkeypatch):
    evals = [
        _eval(f"n{i}", "v0.3.144", "a.pdf", 0.1, f"2026-01-01T00:00:{i:02d}", eval_version="4.3") for i in range(27)
    ]
    evals += [
        _eval(f"m{i}", "v0.3.100", "a.pdf", 0.1, f"2026-01-01T00:00:{i:02d}", eval_version="4.1") for i in range(5)
    ]
    data = _patched_build_data(monkeypatch, evals)
    versions = {o["version"]: o["n"] for o in data["available_eval_versions"]}
    assert versions == {"4.1": 5, "4.3": 27}


def test_available_eval_versions_count_ignores_active_filters(monkeypatch):
    """Dropdown-Optionen zeigen die UNGEFILTERTE Zeilenzahl -- dieselbe
    Konvention wie all_pdfs_opts/all_pvers_opts (Server-Kommentar: 'Dropdown-
    Optionen VOR allen Filtern snapshotten')."""
    evals = [
        _eval(f"a{i}", "v0.3.144", "a.pdf", 0.1, f"2026-01-01T00:00:{i:02d}", eval_version="4.3") for i in range(3)
    ]
    evals += [
        _eval(f"b{i}", "v0.3.144", "b.pdf", 0.1, f"2026-01-02T00:00:{i:02d}", eval_version="4.3") for i in range(4)
    ]
    data = _patched_build_data(monkeypatch, evals, pdf="a")
    versions = {o["version"]: o["n"] for o in data["available_eval_versions"]}
    assert versions["4.3"] == 7  # nicht nur die 3 gefilterten a.pdf-Zeilen


def test_eval_version_query_param_selects_requested_version(monkeypatch):
    evals = [_eval("new", "v0.3.144", "a.pdf", 0.1, "2026-02-01T00:00:00", eval_version="4.3")]
    evals += [_eval("old", "v0.3.100", "a.pdf", 0.2, "2026-01-01T00:00:00", eval_version="4.1")]
    data = _patched_build_data(monkeypatch, evals, eval_version="4.1")
    assert data["eval_version"] == "4.1"
    assert data["pair_matrix"]["eval_version"] == "4.1"


# ── Frontend-Anker (_build_live_html-Muster, kein bash grep/sed auf dem NUL-Byte) ──


def test_html_filterbar_has_eval_version_select():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    filterbar = html[html.index('<div class="filterbar">') : html.index("</div>", html.index('id="global-model"'))]
    section = html[html.index('<div class="filterbar">') : html.index('id="filter-badges"')]
    assert 'id="eval-ver-select"' in section
    assert 'onchange="onEvalVerChange()"' in section
    assert filterbar  # Filterbar-Ausschnitt nicht leer (Sanity)


def test_html_load_and_render_passes_current_eval_version_to_fetch():
    """Bug-Kern: der Fetch reichte eval_version NIE durch (Bestands-Kommentar
    "nie aus URL/State setzen"). Fix muss _currentEvalVersion in die
    URLSearchParams jedes /data.json-Fetches setzen -- gilt auch fuer den
    15s-Poll, da loadAndRender(false) denselben Codepfad nutzt."""
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("async function loadAndRender")
    end = html.index("\n}", html.index("fetch(url)", start))
    block = html[start:end]
    assert "p.set('eval_version', _currentEvalVersion)" in block


def test_html_eval_ver_seeded_from_url_on_boot():
    """_getEvalVerFromUrl() war definiert, aber nirgends aufgerufen (totes
    Deep-Link-Handling). Fix: _currentEvalVersion wird direkt bei der
    Deklaration aus der URL geseedet."""
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    assert "let _currentEvalVersion = _getEvalVerFromUrl();" in html


def test_html_render_with_data_wires_eval_ver_dropdown_from_payload():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    start = html.index("function _renderWithData")
    end = html.index("if (d.all_pvers", start)
    block = html[start:end]
    assert "_initEvalVerDropdown(d.available_eval_versions, d.eval_version)" in block


def test_html_side_foot_shows_persistent_eval_version_pill():
    from generative.eval_dashboard_server import _build_live_html

    html = _build_live_html()
    side_foot = html[html.index('class="side-foot"') : html.index("</aside>")]
    assert 'id="sf-ev-pill"' in side_foot


# ── Sichtpruefungs-Fund 2026-07-16: sf-ev-pill blieb nach manuellem ────────
# Dropdown-Wechsel auf dem alten Wert stehen (Klick-Pfad 4.3 -> 4.1 zeigte
# "eval 4.3" statt "eval 4.1"). Ursache: onEvalVerChange() setzt
# _currentEvalVersion schon VOR dem Fetch auf den neuen Wert -- der
# Post-Fetch-Early-Return in _initEvalVerDropdown ("nichts geaendert, wenn
# Optionsmenge+_currentEvalVersion===selected schon passen") griff dadurch
# faelschlich auch beim ERSTEN Render nach einem echten Wechsel. Test fuehrt
# die echte JS-Funktion in Node aus (kein Nachbau der Logik), simuliert genau
# diese Abfolge: Aufruf 1 (Erstladung 4.3) -> Aufruf 2 mit selected="4.1"
# (Wechsel, _currentEvalVersion vorab auf "4.1" gesetzt wie onEvalVerChange
# es tut) -- die Pill MUSS "4.1" zeigen, nicht "4.3" (stale).


def _extract_js_function(text: str, name: str) -> str:
    start = text.index(f"function {name}(")
    brace_start = text.index("{", start)
    depth = 0
    i = brace_start
    while True:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[start : i + 1]


def test_eval_ver_pill_updates_after_manual_switch_with_unchanged_option_set():
    import json
    import shutil
    import subprocess

    from generative.eval_dashboard_server import _build_live_html

    node = shutil.which("node")
    if node is None:
        pytest.skip("node nicht verfügbar")

    html = _build_live_html()
    fn = _extract_js_function(html, "_initEvalVerDropdown")

    available = [{"version": "4.1", "n": 515}, {"version": "4.3", "n": 27}]
    script = f"""
    // Minimal-DOM-Stub: nur die von _initEvalVerDropdown angefassten Elemente.
    function makeEl() {{ return {{ dataset: {{}}, innerHTML: '', value: '', style: {{}}, textContent: '' }}; }}
    const els = {{
      'eval-ver-select': makeEl(),
      'eval-ver-warn': makeEl(),
      'sf-ev-pill': makeEl(),
    }};
    global.document = {{ getElementById: (id) => els[id] || null }};
    let _currentEvalVersion = null;
    {fn}

    // Aufruf 1: Erstladung, Server liefert Default 4.3.
    _initEvalVerDropdown({json.dumps(available)}, '4.3');
    // Aufruf 2: onEvalVerChange() haette VORHER _currentEvalVersion='4.1' gesetzt
    // (Dropdown-Wert vom Nutzer geaendert) -- Optionsmenge bleibt UNVERAENDERT.
    _currentEvalVersion = '4.1';
    _initEvalVerDropdown({json.dumps(available)}, '4.1');

    process.stdout.write(JSON.stringify({{ pill: els['sf-ev-pill'].textContent, selectValue: els['eval-ver-select'].value }}));
    """
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"node stderr: {result.stderr}"
    out = json.loads(result.stdout)
    assert out["pill"] == "4.1", f"Pill blieb stale: {out}"
    assert out["selectValue"] == "4.1"
