"""eval_dashboard_server.py -- Live-Dashboard-Server fuer Atomic-Agent Eval-Daten.

Startet einen lokalen HTTP-Server auf Port 8051. Das Dashboard:
- Laedt Daten alle 15 Sekunden automatisch neu (kein Browser-Reload)
- Zeigt immer den aktuellen Stand von quality_history.jsonl
- Filtert nach Pipeline-Version und Quell-PDF

Usage: python eval_dashboard_server.py [--port 8050]
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import urlparse, parse_qs

# Importiere alle Daten-Funktionen aus eval_dashboard.py
from generative import eval_dashboard as D

PORT = 8051

# ---------------------------------------------------------------------------
# Daten-Endpunkt
# ---------------------------------------------------------------------------

_AGENT_ORDER = ["planner", "extractor", "verifier", "cross_reference", "critic"]
_AGENT_LABELS = {
    "planner":         "Planner",
    "extractor":       "Extractor",
    "verifier":        "Verifier",
    "cross_reference": "Cross-Ref",
    "critic":          "Critic",
    "eval":            "Eval",
}


def _canonical_agent(name: str) -> str | None:
    """Trace-Agent-Name → Anzeige-Agent; None = nicht als Agent zählen.

    Stage-8-Eval-Judges (eval_quality_*) sind keine Pipeline-Agenten —
    sie werden wie in _live_run_data unter "eval" zusammengefasst.
    """
    if not name or name in ("unknown", "?", "orchestrator"):
        return None
    if name.startswith("eval_quality"):
        return "eval"
    return name


def _is_llm_call_record(r: dict) -> bool:
    """True nur für echte LLM-Call-Records.

    Schema-Invariante (agents/base.py): ``_trace`` schreibt für jeden LLM-Call
    ein ``model``-Feld; ``trace_event`` schreibt für Bookkeeping-Events ein
    ``type``-Feld (verifier ``anchor_stats``, critic ``score_result``,
    orchestrator ``note_outcome``/``plan_stats``) und NIE ein ``model``. Beide
    schließen sich aus → ``model``-Präsenz ist der zuverlässige Diskriminator.

    Ohne diesen Filter zählt z.B. der Verifier „10 calls / 0 Tokens", weil seine
    anchor_stats-Events als Calls mitzählen, obwohl der deterministische
    Pre-Pass gar keinen LLM gerufen hat. Error-Records tragen ``model`` und
    werden bewusst weiter gezählt (fehlgeschlagene Calls sind echte Calls)."""
    return "model" in r

def _read_agent_stats(allowed_run_ids: set | None = None) -> dict:
    """Aggregiert Token- und Dauer-Statistiken je Agent aus runs/*.jsonl.
    Nur Runs der aktuellen (höchsten) Pipeline-Version werden berücksichtigt.
    """
    from collections import defaultdict
    runs_dir = Path(__file__).parent / ".cache" / "runs"
    stats: dict = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0,
                                        "cache_r": 0, "cache_c": 0, "dur_ms": 0, "errors": 0,
                                        "cost_usd": 0.0})
    if not runs_dir.exists():
        return {}
    # Aktuelle Pipeline-Version bestimmen + zugehörige run_ids filtern
    try:
        from generative import db as _db_ag
        db_runs = _db_ag.query_pipeline_runs()
        if db_runs:
            import re as _re
            latest_ver = sorted(
                {r["pipeline_version"] for r in db_runs if r.get("pipeline_version")
                 and not r["pipeline_version"].startswith("extractive-")},
                key=lambda v: [int(x) for x in _re.findall(r"\d+", v)]
            )[-1]
            allowed_ids = {r["run_id"] for r in db_runs if r.get("pipeline_version") == latest_ver}
        else:
            allowed_ids = None
    except Exception:
        allowed_ids = None
    # PDF/Model-Filter überschreibt Version-Filter — explizite run_id-Auswahl hat Vorrang
    if allowed_run_ids is not None:
        allowed_ids = allowed_run_ids
    for f in runs_dir.glob("*.jsonl"):
        if allowed_ids is not None and f.stem not in allowed_ids:
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                a = _canonical_agent(r.get("agent", ""))
                if a is None:
                    continue
                if not _is_llm_call_record(r):
                    continue  # Bookkeeping-Event (anchor_stats etc.) — kein Call
                stats[a]["calls"]   += 1
                stats[a]["input"]   += r.get("input_tokens", 0) or 0
                stats[a]["output"]  += r.get("output_tokens", 0) or 0
                stats[a]["cache_r"] += r.get("cache_read_tokens", 0) or 0
                stats[a]["cache_c"] += r.get("cache_creation_tokens", 0) or 0
                stats[a]["dur_ms"]  += r.get("duration_ms", 0) or 0
                if r.get("error"):
                    stats[a]["errors"] += 1
                try:
                    from generative.config import compute_cost_per_call as _agent_cost_fn
                    stats[a]["cost_usd"] += _agent_cost_fn(
                        model=r.get("model", ""),
                        input_tokens=r.get("input_tokens", 0) or 0,
                        output_tokens=r.get("output_tokens", 0) or 0,
                        cache_read_tokens=r.get("cache_read_tokens", 0) or 0,
                    )
                except Exception:
                    pass
            except Exception:
                pass

    # In sortierte Listenform fuer Chart.js umwandeln
    agents = [a for a in _AGENT_ORDER if a in stats]
    agents += [a for a in sorted(stats) if a not in _AGENT_ORDER]
    labels  = [_AGENT_LABELS.get(a, a) for a in agents]

    def _cache_pct(a):
        total = stats[a]["input"] + stats[a]["output"] + stats[a]["cache_r"]
        return round(stats[a]["cache_r"] / total * 100, 1) if total else 0

    return {
        "agents":      agents,
        "labels":      labels,
        "output":      [stats[a]["output"]            for a in agents],
        "input":       [stats[a]["input"]             for a in agents],
        "cache_r":     [stats[a]["cache_r"]           for a in agents],
        "dur_s":       [round(stats[a]["dur_ms"]/1000,1) for a in agents],
        "calls":       [stats[a]["calls"]             for a in agents],
        "cache_pct":   [_cache_pct(a)                 for a in agents],
        "errors":      [stats[a]["errors"]            for a in agents],
        "cost_usd":    [round(stats[a]["cost_usd"], 4) for a in agents],
    }


def _avg_agreement(rows: list[dict]) -> float | None:
    """Mittelt agreement über Rows; None (nicht 0.0) wenn kein Wert vorliegt —
    sonst rendert das Frontend "0 %" statt "–" bei fehlenden Agreement-Daten."""
    vals = [r["agreement"] for r in rows if r["agreement"] is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def _read_calibration_data(allowed_note_paths: set | None = None,
                           eval_version: str = "4.1") -> dict:
    """Liest Kalibrierungs-Stand: LLM-Eval vs. Human-Labels."""
    try:
        from generative import db as _db
        import sqlite3
        conn = sqlite3.connect(str(_db.DB_PATH))
        conn.row_factory = sqlite3.Row

        # Alle evaluierten Notes — ORDER BY timestamp: bei Mehrfach-Evals
        # derselben Note gewinnt deterministisch die neueste Zeile.
        evaluated = {
            r["note_path"]: {
                "llm_hall": round(r["hallucination_rate"] * 100, 1) if r["hallucination_rate"] is not None and r["hallucination_rate"] >= 0 else None,
                "llm_cov":  round(r["coverage_factual"] * 100, 1) if r["coverage_factual"] is not None and r["coverage_factual"] >= 0 else None,
                "pdf":      r["pdf"],
            }
            for r in conn.execute(
                "SELECT note_path, hallucination_rate, coverage_factual, pdf "
                "FROM note_evals WHERE eval_version=? ORDER BY timestamp",
                (eval_version,),
            ).fetchall()
        }

        # Human-Labels (nach collect.py) — llm_hall_rate ist die zum
        # Label-Zeitpunkt gehoerende LLM-Rate (korrektes Vergleichspaar).
        labeled = {
            r["note_path"]: {
                "human_hall":   round(r["human_hall_rate"] * 100, 1) if r["human_hall_rate"] is not None else None,
                "llm_hall":     round(r["llm_hall_rate"] * 100, 1)   if r["llm_hall_rate"]   is not None else None,
                "agreement":    round(r["agreement_rate"] * 100, 1)  if r["agreement_rate"]  is not None else None,
                "n_claims":     r["n_claims"],
                "n_supported":  r["n_supported"],
                "n_hallucinated": r["n_hallucinated"],
            }
            for r in conn.execute(
                "SELECT note_path, human_hall_rate, llm_hall_rate, agreement_rate, "
                "n_claims, n_supported, n_hallucinated "
                "FROM calibration_labels WHERE eval_version=?",
                (eval_version,),
            ).fetchall()
        }
        conn.close()

        rows = []
        for note, ev in evaluated.items():
            if allowed_note_paths is not None and note not in allowed_note_paths:
                continue
            lab = labeled.get(note, {})
            status = "labeled" if note in labeled else "evaluated"
            # Gelabelte Notes: llm_hall aus dem Label-Paar, nicht aus der
            # (womoeglich anderen) letzten note_evals-Zeile.
            llm_hall = lab.get("llm_hall") if status == "labeled" else ev.get("llm_hall")
            rows.append({
                "note":       note.replace("vault__", "").replace(".md", ""),
                "pdf":        ev.get("pdf", ""),
                "llm_hall":   llm_hall,
                "llm_cov":    ev.get("llm_cov"),
                "human_hall": lab.get("human_hall"),
                "agreement":  lab.get("agreement"),
                "n_claims":   lab.get("n_claims"),
                "status":     status,
            })

        # Arbeitslisten-Sortierung: ungelabelte zuerst ("gelabelt" = Zeile in
        # calibration_labels existiert — collect.py schreibt atomar komplett),
        # darin nach LLM-Fehlerquote absteigend (groesster Informationsgewinn
        # pro Label zuerst), None-Raten ans Ende. Gelabelte danach.
        rows.sort(key=lambda r: (
            r["status"] == "labeled",
            -(r["llm_hall"] if r["llm_hall"] is not None else -1),
        ))

        # Zaehler aus den gefilterten rows — sonst zeigt der Strip die
        # Gesamtmenge, waehrend die Tabelle gefiltert ist.
        n_eval    = len(rows)
        n_labeled = sum(1 for r in rows if r["status"] == "labeled")
        avg_agree = _avg_agreement(rows)

        return {
            "rows":      rows,
            "n_eval":    n_eval,
            "n_labeled": n_labeled,
            "avg_agree": avg_agree,
            "has_labels": n_labeled > 0,
        }
    except Exception:
        return {"rows": [], "n_eval": 0, "n_labeled": 0, "avg_agree": None, "has_labels": False}


def _available_eval_versions(rows: list[dict]) -> list[str]:
    """Gibt sortierte Liste aller eval_versions zurueck."""
    def _vkey(v: str) -> tuple:
        import re
        parts = re.findall(r"\d+", v)
        return tuple(int(p) for p in parts) if parts else (0,)
    seen = sorted(set(r["eval_version"] for r in rows if r.get("eval_version")), key=_vkey)
    return seen


def build_data(eval_version: str | None = None,
               pipeline_version: str | None = None,
               language: str | None = None,
               model: str | None = None,
               pdf: str | None = None) -> dict:
    """Baut alle Chart-Daten aus SQLite (note_evals) + Log-Files (runs).

    eval_version:      wenn None → neueste verfuegbare Version
    pipeline_version:  wenn None → alle Versionen
    """
    from generative import db as _db
    # Qualitaets-Daten aus SQLite (primaer) mit Fallback auf JSONL
    try:
        all_quality_rows = _db.query_note_evals()
        if not all_quality_rows:
            raise ValueError("DB leer")
    except Exception:
        all_quality_rows = D._read_quality_history()

    available_versions = _available_eval_versions(all_quality_rows)

    # Default: neueste Version
    if eval_version is None or eval_version not in available_versions:
        eval_version = available_versions[-1] if available_versions else None

    # Filter — niemals Versionen mischen
    quality_rows = [r for r in all_quality_rows if r.get("eval_version") == eval_version] if eval_version else []
    # Dropdown-Optionen VOR dem Filtern — immer alle Optionen anzeigen
    all_langs   = sorted({r.get("language") or "" for r in quality_rows if r.get("language")})
    if pipeline_version:
        quality_rows = [r for r in quality_rows
                        if (r.get("version") or r.get("pipeline_version")) == pipeline_version]
    if language:
        quality_rows = [r for r in quality_rows if r.get("language") == language]
    # Merken: all_langs bleibt unverändert (alle Sprachen, nicht nur die gefilterte)
    if pdf:
        # pdf-Wert aus Dropdown ist lowercase Label (z.B. "afzal")
        # quality_rows.pdf kann voller Dateiname sein ("Afzal - 2017 - ....pdf") → startswith-Match
        quality_rows = [r for r in quality_rows
                        if (r.get("pdf") or "").lower().startswith(pdf.lower())]

    all_log_runs = D._read_all_log_runs()
    token_runs   = D._read_token_runs()
    # pdf_label + pipeline_version aus pipeline_runs DB in token_runs eintragen
    try:
        from generative import db as _db_tok
        _run_info = {r["run_id"]: {"pdf_label": r.get("pdf_label") or r.get("pdf_source",""),
                                    "ver":       r.get("pipeline_version") or "",
                                    "model":     r.get("model") or "",
                                    "cost_usd":  r.get("cost_usd", 0.0) or 0.0}
                     for r in _db_tok.query_pipeline_runs()}
        for tr in token_runs:
            info = _run_info.get(tr.get("run_id",""), {})
            tr["pdf_label"] = info.get("pdf_label","")
            tr["ver"]       = info.get("ver","")
            tr["model"]     = info.get("model","")
            tr["cost_usd"]  = info.get("cost_usd", 0.0)
        # model aus JSONL-Traces backfillen wenn DB-Eintrag leer
        runs_dir = Path(__file__).parent / ".cache" / "runs"
        for tr in token_runs:
            if not tr["model"] and tr.get("run_id"):
                jl = runs_dir / f"{tr['run_id']}.jsonl"
                if jl.exists():
                    import json as _json2
                    for line in jl.read_text(encoding="utf-8", errors="replace").splitlines()[:5]:
                        try:
                            m = _json2.loads(line.strip()).get("model","")
                            if m: tr["model"] = m; break
                        except Exception:
                            pass
    except Exception:
        pass
    # run_id → Sprache aus note_evals (jeder Run hat konsistente Sprache)
    _run_lang: dict[str, str] = {}
    try:
        for r in all_quality_rows:
            rid = r.get("run_id","")
            if rid and r.get("language") and rid not in _run_lang:
                _run_lang[rid] = r.get("language","")
        for tr in token_runs:
            tr["language"] = _run_lang.get(tr.get("run_id",""), "")
    except Exception:
        pass

    # DB-only Runs (z.B. extractive — kein JSONL) zu token_runs hinzufuegen
    try:
        from generative import db as _db_only
        _jsonl_run_ids = {tr.get("run_id") for tr in token_runs}
        _run_lang_safe = _run_lang if "_run_lang" in dir() else {}
        for r in _db_only.query_pipeline_runs():
            if r["run_id"] not in _jsonl_run_ids and r.get("model"):
                token_runs.append({
                    "run_id":       r["run_id"],
                    "model":        r.get("model", ""),
                    "ver":          r.get("pipeline_version", ""),
                    "pdf_label":    r.get("pdf_label") or r.get("pdf_source", ""),
                    "language":     _run_lang_safe.get(r["run_id"], ""),
                    "cost_usd":     r.get("cost_usd", 0.0) or 0.0,
                    "tokens_in":    0,
                    "tokens_out":   0,
                    "tokens_cache": 0,
                    "duration_min": round((r.get("duration_s") or 0.0) / 60, 1),
                    "calls":        0,
                    "date":         "",
                })
    except Exception as _e:
        import sys as _sys; print(f"[db-only-runs] Fehler: {_e}", file=_sys.stderr)

    # ── Dropdown-Optionen VOR allen Filtern snapshotten ──────────────
    # Smoke-/Test-Modelle aus dem Filter halten (z. B. "m", "test", "smoke-model").
    _MODEL_DENYLIST = {"m", "test", "smoke-model"}
    _all_models_opts = sorted({m for tr in token_runs
                               if (m := tr.get("model", ""))
                               and m not in _MODEL_DENYLIST and "smoke" not in m.lower()})

    # ── token_runs + quality_rows + all_log_runs gemeinsam filtern ──────
    # Alle Filter auf token_runs anwenden → run_ids extrahieren → quality_rows + log_runs ebenfalls filtern
    if model:
        token_runs = [tr for tr in token_runs if tr.get("model") == model]
    if pipeline_version:
        token_runs = [tr for tr in token_runs if tr.get("ver") == pipeline_version]
    if language:
        token_runs = [tr for tr in token_runs if tr.get("language") == language]
    if pdf:
        token_runs = [tr for tr in token_runs
                      if (tr.get("pdf_label","")).lower().startswith(pdf.lower())]

    # run_ids der gefilterten token_runs → quality_rows auf selbe Runs einschränken
    # Wenn Filter aktiv aber keine run_ids matchen → quality_rows leeren (nicht überspringen)
    if model or pipeline_version:
        _filtered_run_ids = {tr.get("run_id") for tr in token_runs if tr.get("run_id")}
        quality_rows = [r for r in quality_rows if r.get("run_id") in _filtered_run_ids]

    log_data     = D._build_log_data(all_log_runs)

    # Wenn keine Log-Runs: pipeline_runs aus SQLite als Fallback
    if not all_log_runs:
        from generative import db as _db2
        db_runs = _db2.query_pipeline_runs()
        all_log_runs = []
        for r in db_runs:
            if not r.get("pipeline_version"):
                continue
            n_gen = r.get("n_generated", 0) or 0
            n_vlt = r.get("n_vault", 0) or 0
            key   = r.get("pdf_key") or (r.get("pdf_source") or "").split(".")[0].split(" - ")[0].strip().lower()
            label = r.get("pdf_label") or (r.get("pdf_source") or "").split(".")[0].split(" - ")[0].strip()
            all_log_runs.append({
                "run_id":     r.get("run_id", ""),
                "model":      r.get("model", ""),
                "ver":        r.get("pipeline_version"),
                "key":        key,
                "label":      label,
                "n_total":    n_gen,
                "n_vault":    n_vlt,
                "n_inbox":    r.get("n_inbox", 0) or 0,
                "n_merge":    r.get("n_merge", 0) or 0,
                "n_dropped":  r.get("n_dropped", 0) or 0,
                "n_words":    r.get("n_words", 0) or 0,
                "words":      r.get("n_words") or None,
                "pages":      0,
                "accept_pct": round(n_vlt / n_gen * 100, 1) if n_gen > 0 else 0.0,
            })
        log_data = D._build_log_data(all_log_runs)

    # ── all_log_runs Dropdown-Optionen VOR all_log_runs-Filtern ──────
    # foss nur im ungefilterten Default-View ausblenden — sobald ein Modell-/
    # Versions-Filter aktiv ist, bleibt foss einsehbar (#36, gleiche Bedingung
    # wie der foss-Ausschluss aus quality_rows/all_log_runs unten).
    _exclude_foss = not (model or pipeline_version)

    # PDF-Dropdown: aus den Eval-Daten der aktiven eval_version statt aus
    # all_log_runs — so erscheinen nur PDFs mit echten Daten im View (sonst
    # listet das Dropdown z. B. foss-only evaluierte PDFs, die „0 Notes" ergeben,
    # solange kein foss-Modell gewählt ist). Volltitel, dedupliziert.
    _all_pdfs_opts  = D._dedupe_pdf_options(
        r.get("pdf") for r in all_quality_rows
        if r.get("eval_version") == eval_version and r.get("pdf")
        and not (_exclude_foss
                 and D.is_foss_version(r.get("version") or r.get("pipeline_version") or "")))
    # Versions-Dropdown: nur Pipeline-Versionen MIT Eval-Daten in der aktiven
    # eval_version, eingeschränkt auf den aktiven PDF-/Sprach-Filter. So
    # verschwinden sowohl Alt-Versionen ohne 4.1-Daten als auch Versionen, die
    # die gewählte PDF nie evaluiert haben (statt 61 log-basierte Versionen).
    _pver_rows = [r for r in all_quality_rows if r.get("eval_version") == eval_version]
    if language:
        _pver_rows = [r for r in _pver_rows if r.get("language") == language]
    if pdf:
        _pver_rows = [r for r in _pver_rows
                      if (r.get("pdf") or "").lower().startswith(pdf.lower())]
    _pver_counts: dict[str, int] = {}
    for r in _pver_rows:
        pv = r.get("version") or r.get("pipeline_version")
        if pv and not (_exclude_foss and D.is_foss_version(pv)):
            _pver_counts[pv] = _pver_counts.get(pv, 0) + 1
    if pdf:
        # PDF-Filter aktiv: alle Versionen mit dieser PDF zeigen — pro PDF ist n
        # naturgemäß klein, eine n≥3-Schwelle würde fast alles ausblenden.
        _all_pvers_opts = sorted(_pver_counts, key=D._ver_sort_key, reverse=True)
    else:
        # Default: 15 neueste Versionen mit echter Stichprobe (n≥3), die
        # allerneueste immer dabei (sonst füllen Einzel-Note-Läufe den Filter).
        _all_pvers_opts = D._top_versions(_pver_counts, limit=15, min_n=3)

    # PDF + Language + Version + Model-Filter auf all_log_runs (nach DB-Fallback)
    if model:
        all_log_runs = [r for r in all_log_runs if r.get("model") == model]
        log_data = D._build_log_data(all_log_runs)
    if pipeline_version:
        all_log_runs = [r for r in all_log_runs if r.get("ver") == pipeline_version]
        log_data = D._build_log_data(all_log_runs)
    if pdf:
        all_log_runs = [r for r in all_log_runs
                        if (r.get("label") or r.get("key","")).lower().startswith(pdf.lower())]
        log_data = D._build_log_data(all_log_runs)
    if language:
        # pdf_label → sprache aus token_runs (nach language-Filter bereits korrekt gefiltert)
        _lang_pdfs = {tr.get("pdf_label","").lower() for tr in token_runs if tr.get("pdf_label")}
        if _lang_pdfs:
            all_log_runs = [r for r in all_log_runs
                            if (r.get("label","")).lower() in _lang_pdfs
                            or (r.get("key","")).lower() in _lang_pdfs]
            log_data = D._build_log_data(all_log_runs)

    # foss-Pipeline (gliner/extractive) nicht mit generativer mischen:
    # im ungefilterten Default-View foss ausschliessen — ueber Modell-/Versions-
    # Filter bleibt foss einsehbar (#36, User-Wunsch 2026-06-19)
    if _exclude_foss:
        all_log_runs = [r for r in all_log_runs if not D.is_foss_version(r.get("ver"))]
        log_data = D._build_log_data(all_log_runs)
        quality_rows = [r for r in quality_rows
                        if not D.is_foss_version(r.get("version") or r.get("pipeline_version"))]
        token_runs = [tr for tr in token_runs if not D.is_foss_version(tr.get("ver"))]

    # Log-Runs nach Version gruppiert
    runs_by_version: dict = {}
    for r in all_log_runs:
        ver = r["ver"]
        if ver not in runs_by_version:
            runs_by_version[ver] = {
                "n_runs": 0, "n_total": 0, "n_vault": 0,
                "n_merge": 0, "n_inbox": 0, "n_dropped": 0, "pdfs": set()
            }
        runs_by_version[ver]["n_runs"]  += 1
        runs_by_version[ver]["n_total"] += r["n_total"]
        runs_by_version[ver]["n_vault"] += r["n_vault"]
        runs_by_version[ver]["n_merge"]   += r.get("n_merge", 0)
        runs_by_version[ver]["n_inbox"]   += r.get("n_inbox", 0)
        runs_by_version[ver]["n_dropped"] += r.get("n_dropped", 0)
        runs_by_version[ver]["pdfs"].add(r["key"])
    # Sets zu Listen konvertieren (JSON-serialisierbar)
    for ver in runs_by_version:
        runs_by_version[ver]["pdfs"] = list(runs_by_version[ver]["pdfs"])

    # Qualitätsmetriken nach Pipeline-Version gruppiert (Median + Mean)
    import statistics
    def _safe_median(lst):
        return round(statistics.median(lst), 1) if lst else None

    quality_by_version: dict = {}
    for r in quality_rows:
        ver = r.get("version") or r.get("pipeline_version") or "unbekannt"
        if ver not in quality_by_version:
            quality_by_version[ver] = {"hall": [], "cov": [], "accept": [], "n": 0, "rows": []}
        quality_by_version[ver]["rows"].append(r)
        hall_val = r.get("hallucination_rate")
        if hall_val is not None and float(hall_val) >= 0:
            quality_by_version[ver]["hall"].append(float(hall_val) * 100)
        cov = r.get("coverage_factual") or r.get("coverage_rate")
        if cov is not None and float(cov) >= 0:
            quality_by_version[ver]["cov"].append(float(cov) * 100)
        quality_by_version[ver]["n"] += 1

    # Statistiken berechnen (Median ist primär, Mean sekundär)
    def _vkey(v):
        import re; parts = re.findall(r"\d+", v); return tuple(int(p) for p in parts) if parts else (0,)
    sorted_pipeline_versions = sorted(quality_by_version.keys(), key=_vkey)

    for ver, d2 in quality_by_version.items():
        # avg_hall = gepoolte Rate (ankergewichtet, Mean-Fallback) — identische
        # Definition wie die KPI-Kachel in _calc_kpis, damit Kachel und Sparkline
        # denselben Wert zeigen.
        d2["avg_hall"]    = D._pooled_hall_pct(d2["rows"])
        d2["avg_cov"]     = round(sum(d2["cov"])  / len(d2["cov"]),  1) if d2["cov"]  else None
        d2["median_hall"] = _safe_median(d2["hall"])
        d2["median_cov"]  = _safe_median(d2["cov"])

    # Trend-Daten fuer KPI-Drill-Down (sortierte Listen parallel zu sorted_pipeline_versions)
    # Akzeptanzrate je Pipeline-Version: gepoolt (sum vault / sum total) —
    # gleiche Metrik-Definition wie die KPI-Kachel in _calc_kpis, sonst
    # zeigen Kachel und Sparkline verschiedene Werte.
    accept_pairs_by_ver: dict[str, list[tuple[int, int]]] = {}
    for r in all_log_runs:
        ver = r.get("ver") or "?"
        accept_pairs_by_ver.setdefault(ver, []).append(
            (r.get("n_vault", 0) or 0, r.get("n_total", 0) or 0))

    def _pooled_accept(ver: str) -> float | None:
        pairs = accept_pairs_by_ver.get(ver)
        if not pairs:
            return None
        total = sum(t for _, t in pairs)
        return round(sum(v for v, _ in pairs) / total * 100, 1) if total else None

    # Laufzeit + Tokens je Pipeline-Version (aus Token-Runs)
    dur_by_ver:  dict[str, list[float]] = {}
    tok_by_ver:  dict[str, list[float]] = {}
    cost_by_ver: dict[str, list[float]] = {}
    for r in token_runs:
        ver = r.get("ver") or r.get("pipeline_version") or "?"
        # token_runs hat duration_min (nicht wall_time_s/duration_s)
        dur_min = r.get("duration_min") or 0
        if dur_min:
            dur_by_ver.setdefault(ver, []).append(round(dur_min, 1))
        # token_runs hat tokens_in/tokens_out (nicht tokens_input/tokens_output)
        tok = (r.get("tokens_in", 0) or 0) + (r.get("tokens_out", 0) or 0)
        if tok:
            tok_by_ver.setdefault(ver, []).append(round(tok / 1000, 1))  # in k-Tokens
        cost = r.get("cost_usd", 0.0) or 0.0
        if cost > 0:
            cost_by_ver.setdefault(ver, []).append(cost)

    kpi_trend = {
        "versions": sorted_pipeline_versions,
        # hall: Mean (avg_hall), passend zur KPI-Kachel in _calc_kpis — die
        # zero-inflated Verteilung lässt den Median sonst auf 0 kollabieren.
        # cov bleibt Median (nicht zero-inflated, robuster Lagewert).
        "hall":     [quality_by_version[v].get("avg_hall")   for v in sorted_pipeline_versions],
        "cov":      [quality_by_version[v].get("median_cov")  for v in sorted_pipeline_versions],
        "n":        [quality_by_version[v]["n"]               for v in sorted_pipeline_versions],
        "accept":   [_pooled_accept(v) for v in sorted_pipeline_versions],
        "dur":      [round(sum(dur_by_ver.get(v,[])) / len(dur_by_ver[v]), 1) if dur_by_ver.get(v) else None for v in sorted_pipeline_versions],
        "tokens":   [round(sum(tok_by_ver.get(v,[])) / 1000, 1) if tok_by_ver.get(v) else None for v in sorted_pipeline_versions],  # in M-Tokens
        "cost":     [round(sum(cost_by_ver.get(v, [])), 4) if cost_by_ver.get(v) else None for v in sorted_pipeline_versions],
    }
    # Delta neueste-vs-Vorversion pro KPI (mit N-Guard, #36 P4)
    kpi_trend["deltas"] = {
        m: D.version_delta(kpi_trend, m)
        for m in ("hall", "cov", "n", "accept", "dur", "tokens", "cost")
    }

    return {
        "generated_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_version":        eval_version,
        "available_eval_versions": available_versions,
        "kpis":                D._calc_kpis(log_data, all_log_runs, quality_rows, token_runs),
        "pdf_table":           D._calc_pdf_table(log_data, all_log_runs, quality_rows),
        "accept":              D._chart_acceptance(log_data),
        "scatter":             _chart_scatter_versioned(quality_rows),
        "long":                D._chart_longitudinal(log_data),
        "tokens":              D._chart_tokens_by_version(token_runs),
        "scaling":             D._chart_scaling(all_log_runs),
        "quality_by_version":  quality_by_version,
        "runs_by_version":     runs_by_version,
        "pdf_words":           {r["label"]: r["n_words"] for r in all_log_runs if r.get("n_words", 0) > 0
                                } | {r["key"]: r["n_words"] for r in all_log_runs if r.get("n_words", 0) > 0},
        "kpi_trend":           kpi_trend,
        "all_langs":           all_langs,
        "all_models":          _all_models_opts,
        "all_pvers":           _all_pvers_opts,
        "all_pdfs":            _all_pdfs_opts,
        "agent_stats":         _read_agent_stats(
                                   allowed_run_ids={tr.get("run_id") for tr in token_runs if tr.get("run_id")}
                                   if (pdf or model or language) else None
                               ),
        "calibration":         _read_calibration_data(
                                   allowed_note_paths={r.get("note_path") or r.get("note") for r in quality_rows}
                                   if (pdf or language or model or pipeline_version) else None,
                                   eval_version=eval_version or "4.1",
                               ),
        "pdf_meta":            {k: v for k, v in D._PDF_META.items()},
        "vault_name":          _vault_name(),
        "thresholds": {
            "accept": list(D.THRESH_ACCEPT),
            "hall":   list(D.THRESH_HALL),
            "cov":    list(D.THRESH_COV),
        },
    }


def _vault_name() -> str:
    """Vault-Name fuer obsidian://-Links — aus der Config, nie hartcodiert."""
    try:
        from generative.config import VAULT
        return Path(VAULT).name
    except Exception:
        return ""


def _chart_scatter_versioned(quality_rows: list[dict]) -> dict:
    """Scatter-Daten mit Version-Info fuer den Version-Filter."""
    points:   list[dict] = []
    pdf_map:  dict[str, str] = {}
    versions: list[str] = []

    import re as _re
    for r in quality_rows:
        hall = r.get("hallucination_rate")
        cov  = r.get("coverage_factual") or r.get("coverage_rate")
        # Sentinel-Werte (-1.0 = ungültig) überspringen
        if hall is None or cov is None or float(hall) < 0 or float(cov) < 0:
            continue
        label = r.get("note_path") or r.get("note") or r.get("note_title") or "?"
        label = _re.sub(r"^(vault|inbox)__", "", label).replace(".md", "")
        pdf   = r.get("pdf") or r.get("source_pdf") or "unbekannt"
        ver   = r.get("version") or r.get("pipeline_version") or "unbekannt"
        if pdf not in pdf_map:
            pdf_map[pdf] = D._pdf_short_name(pdf)
        if ver not in versions:
            versions.append(ver)
        points.append({
            "x":         round(float(hall) * 100, 1),
            "y":         round(float(cov)  * 100, 1),
            "label":     label,
            "pdf":       pdf,
            "pdf_label": pdf_map[pdf],
            "version":   ver,
            # Drill-Down-Drawer: Identifikation + Refresh-Persistenz-Key
            "run_id":       r.get("run_id", ""),
            "eval_version": r.get("eval_version", ""),
        })

    pdfs = [{"raw": k, "label": v} for k, v in pdf_map.items()]
    # Versionen nach Versionsnummer sortieren
    def _vkey(v: str) -> tuple:
        import re
        return tuple(int(n) for n in re.findall(r"\d+", v))
    versions_sorted = sorted(versions, key=_vkey)
    return {"points": points, "pdfs": pdfs, "versions": versions_sorted}


# ---------------------------------------------------------------------------
# Live-Monitor
# ---------------------------------------------------------------------------

_STAGE_ORDER = ["planner", "extractor", "llm_dedup", "verifier", "cross_reference", "critic", "eval"]
_STAGE_LABELS = {
    "planner":         "Planner",
    "extractor":       "Extractor",
    "llm_dedup":       "ER-Dedup",
    "verifier":        "Verifier",
    "cross_reference": "Cross-Ref",
    "critic":          "Critic",
    "eval":            "Eval",
}


def _live_run_data() -> dict:
    """Liest die neueste runs/*.jsonl und gibt Fortschritts-Aggregat zurück."""
    import time
    from collections import defaultdict
    runs_dir = Path(__file__).parent / ".cache" / "runs"
    if not runs_dir.exists():
        return {"run_id": None, "is_running": False, "agents": {}}

    files = sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {"run_id": None, "is_running": False, "agents": {}}

    latest = files[-1]
    age_s = time.time() - latest.stat().st_mtime
    is_running = age_s < 120  # kein neuer Eintrag seit >120s → wahrscheinlich fertig

    agents: dict = defaultdict(lambda: {"calls": 0, "cached": 0,
                                         "tokens_in": 0, "tokens_out": 0,
                                         "dur_ms": 0, "errors": 0})
    first_ts = last_ts = last_agent = None

    for line in latest.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        agent = e.get("agent", "unknown")
        # Eval-Agents zusammenfassen
        if agent.startswith("eval_quality"):
            agent = "eval"
        # Nur echte LLM-Calls in die Per-Agent-Stats (Bookkeeping-Events wie
        # anchor_stats tragen kein model/keine Tokens). Timing/last-activity
        # unten gilt aber für ALLE Events — sonst zeigt die Live-Ansicht einen
        # zu frühen „letzten Schritt", wenn das letzte Event Bookkeeping war.
        if _is_llm_call_record(e):
            a = agents[agent]
            a["calls"] += 1
            a["cached"] += 1 if e.get("cached") else 0
            a["tokens_in"]  += e.get("input_tokens", 0)
            a["tokens_out"] += e.get("output_tokens", 0)
            a["dur_ms"]     += e.get("duration_ms", 0)
            a["errors"]     += 1 if e.get("error") else 0
        ts = e.get("ts", "")
        if not first_ts:
            first_ts = ts
        last_ts = ts
        last_agent = agent

    # Elapsed aus Timestamps
    elapsed_s = 0
    if first_ts and last_ts:
        try:
            t0 = datetime.fromisoformat(first_ts)
            t1 = datetime.fromisoformat(last_ts)
            elapsed_s = int((t1 - t0).total_seconds())
        except Exception:
            pass

    total_in  = sum(v["tokens_in"]  for v in agents.values())
    total_out = sum(v["tokens_out"] for v in agents.values())

    return {
        "run_id":     latest.stem,
        "is_running": is_running,
        "started_at": first_ts,
        "last_ts":    last_ts,
        "last_agent": last_agent,
        "elapsed_s":  elapsed_s,
        "agents":     dict(agents),
        "total":      {"tokens_in": total_in, "tokens_out": total_out,
                       "calls": sum(v["calls"] for v in agents.values())},
    }


_LIVE_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Atomic Agent — Live Monitor</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
    --ink: #e2e4ed; --muted: #7c7f96; --accent: #6c8ef5;
    --green: #4ade80; --amber: #fbbf24; --red: #f87171;
    --font: 'IBM Plex Mono', 'Fira Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--ink); font-family: var(--font);
         font-size: 13px; padding: 24px; }
  h1 { font-size: 16px; font-weight: 600; letter-spacing: .05em;
       color: var(--accent); margin-bottom: 4px; }
  .meta { color: var(--muted); font-size: 11px; margin-bottom: 20px; }
  .pulse { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           background: var(--green); margin-right: 6px;
           animation: pulse 1.2s ease-in-out infinite; }
  .pulse.idle { background: var(--muted); animation: none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
          margin-bottom: 20px; }
  .kpi { background: var(--card); border: 1px solid var(--border);
         border-radius: 8px; padding: 14px 16px; }
  .kpi .label { color: var(--muted); font-size: 10px; text-transform: uppercase;
                letter-spacing: .08em; margin-bottom: 4px; }
  .kpi .value { font-size: 22px; font-weight: 700; color: var(--ink); }
  .kpi .sub   { color: var(--muted); font-size: 11px; margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--muted); font-size: 10px;
       text-transform: uppercase; letter-spacing: .08em;
       padding: 6px 10px; border-bottom: 1px solid var(--border); }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .bar-wrap { background: var(--border); border-radius: 4px;
              height: 6px; width: 120px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 4px; background: var(--accent);
               transition: width .4s; }
  .bar-fill.cached { background: var(--amber); }
  .stage-done   { color: var(--green); }
  .stage-active { color: var(--accent); font-weight: 600; }
  .stage-wait   { color: var(--border); }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; padding: 16px; }
  .section-title { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
                   color: var(--muted); margin-bottom: 12px; }
</style>
</head>
<body>
<h1>⚡ Atomic Agent — Live Monitor</h1>
<div class="meta" id="meta">Lade…</div>

<div class="grid">
  <div class="kpi"><div class="label">Elapsed</div>
    <div class="value" id="elapsed">—</div>
    <div class="sub" id="run-id">—</div></div>
  <div class="kpi"><div class="label">Tokens gesamt</div>
    <div class="value" id="tokens-total">—</div>
    <div class="sub" id="tokens-sub">in / out</div></div>
  <div class="kpi"><div class="label">LLM-Calls</div>
    <div class="value" id="calls-total">—</div>
    <div class="sub" id="calls-sub">—</div></div>
</div>

<div class="card">
  <div class="section-title">Stage-Fortschritt</div>
  <table id="agent-table">
    <thead><tr>
      <th>Agent</th><th>Calls</th><th>Tokens In</th><th>Tokens Out</th>
      <th>Cached</th><th>Ø Dauer</th>
    </tr></thead>
    <tbody id="agent-body"></tbody>
  </table>
</div>

<script>
const STAGE_ORDER = ["planner","extractor","llm_dedup","verifier","cross_reference","critic","eval"];
const STAGE_LABELS = {
  planner:"Planner", extractor:"Extractor", llm_dedup:"ER-Dedup",
  verifier:"Verifier", cross_reference:"Cross-Ref", critic:"Critic", eval:"Eval"
};

function fmt(n) {
  if (n >= 1000000) return (n/1000000).toFixed(1)+'M';
  if (n >= 1000)    return (n/1000).toFixed(1)+'k';
  return n.toString();
}
function fmtTime(s) {
  if (s < 60) return s + 's';
  return Math.floor(s/60) + 'm ' + (s%60) + 's';
}

async function refresh() {
  try {
    const r = await fetch('/live.json');
    const d = await r.json();

    const running = d.is_running;
    document.getElementById('meta').innerHTML =
      `<span class="pulse ${running?'':'idle'}"></span>` +
      (running ? `Läuft — letzter Event: ${d.last_agent || '—'} @ ${(d.last_ts||'').slice(11,19)}`
               : `Fertig — ${(d.last_ts||'').slice(0,19)}`);

    document.getElementById('elapsed').textContent = d.elapsed_s ? fmtTime(d.elapsed_s) : '—';
    document.getElementById('run-id').textContent = d.run_id || '—';

    const tot = d.total || {};
    document.getElementById('tokens-total').textContent = fmt((tot.tokens_in||0)+(tot.tokens_out||0));
    document.getElementById('tokens-sub').textContent =
      `↑${fmt(tot.tokens_in||0)} ↓${fmt(tot.tokens_out||0)}`;
    document.getElementById('calls-total').textContent = tot.calls || '—';

    const agents = d.agents || {};
    const seenAgents = new Set(Object.keys(agents));
    const allStages = [...STAGE_ORDER, ...Object.keys(agents).filter(a => !STAGE_ORDER.includes(a))];
    const lastAgent = d.last_agent;

    let rows = '';
    for (const s of allStages) {
      const a = agents[s];
      if (!a) continue;
      const label = STAGE_LABELS[s] || s;
      const isLast = s === lastAgent && d.is_running;
      const cls = d.is_running && isLast ? 'stage-active' : (a ? 'stage-done' : 'stage-wait');
      const avgDur = a.calls ? Math.round(a.dur_ms / a.calls / 1000) : 0;
      const cachedPct = a.calls ? Math.round(a.cached / a.calls * 100) : 0;
      rows += `<tr>
        <td class="${cls}">${isLast ? '▶ ' : ''}${label}</td>
        <td>${a.calls}</td>
        <td>${fmt(a.tokens_in)}</td>
        <td>${fmt(a.tokens_out)}</td>
        <td>
          <div class="bar-wrap"><div class="bar-fill cached" style="width:${cachedPct}%"></div></div>
          <span style="color:var(--muted);font-size:10px">${cachedPct}%</span>
        </td>
        <td style="color:var(--muted)">${avgDur}s</td>
      </tr>`;
    }
    document.getElementById('agent-body').innerHTML = rows;
  } catch(e) {
    document.getElementById('meta').textContent = 'Fehler: ' + e.message;
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP-Handler
# ---------------------------------------------------------------------------

def _get_html() -> str:
    return _build_live_html()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # kein Server-Log im Terminal

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/data.json":
            eval_ver  = params.get("eval_version",      [None])[0]
            pver      = params.get("pipeline_version",  [None])[0]
            lang      = params.get("language",           [None])[0]
            mdl       = params.get("model",              [None])[0]
            pdf_f     = params.get("pdf",                [None])[0]
            try:
                data = build_data(eval_version=eval_ver, pipeline_version=pver,
                                  language=lang, model=mdl, pdf=pdf_f)
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())

        elif parsed.path in ("/", "/index.html"):
            body = _get_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/live.json":
            try:
                body = json.dumps(_live_run_data(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())

        elif parsed.path == "/live":
            body = _LIVE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()



# ---------------------------------------------------------------------------
# HTML-Template (ohne eingebettete Daten -- laedt per fetch)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE_PATH = Path(__file__).parent.parent / "internal" / "dashboard" / "eval_dashboard.html"


def _build_live_html() -> str:
    """Liest das Dashboard-HTML und setzt den Port in den Footer ein.

    Das HTML ist live-only (laedt via fetch('/data.json')); der fruehere
    Mock-Pfad (data.js/MOCK_DATA) wurde 2026-06-10 entfernt — ebenso die
    String-Replace-Patches hier, die teils laengst ins Leere liefen.
    """
    html = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace(
        '<span>atomic agent <span class="sep">·</span> pipeline eval <span class="sep">·</span> localhost</span>',
        '<span>atomic agent <span class="sep">·</span> pipeline eval <span class="sep">·</span> localhost:__PORT__</span>'
    )
    return html.replace('__PORT__', str(PORT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"[dashboard] Server laeuft auf {url}")
    print(f"[dashboard] Daten werden alle 15 Sekunden automatisch aktualisiert.")
    print(f"[dashboard] Beenden mit Ctrl+C")

    Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Server gestoppt.")


if __name__ == "__main__":
    main()
