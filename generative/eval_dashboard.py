"""eval_dashboard.py -- Interaktives HTML-Dashboard Für Atomic-Agent Eval-Daten.

Liest:
  .cache/quality_history.jsonl    -- Stage-8-Eval pro Note
  .cache/eval/baseline/*.log      -- DRY-RUN Vault/Inbox-Routing-Logs
  .cache/runs/*.jsonl             -- Token- und Zeittracking pro LLM-Call

Schreibt: .cache/eval/dashboard.html  ->  oeffnet im Browser.

Usage: python eval_dashboard.py
"""
from __future__ import annotations

import json
import re
import statistics
import webbrowser
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / ".cache"
EVAL_DIR  = CACHE_DIR / "eval" / "baseline"
RUNS_DIR  = CACHE_DIR / "runs"
QUALITY_HISTORY = CACHE_DIR / "quality_history.jsonl"
OUTPUT = CACHE_DIR / "eval" / "dashboard.html"

_NOTE_RE    = re.compile(r"^\s*\[DRY-RUN\] -> (Vault|Inbox)[^:]*: (.+?)\.md\b")
_MERGE_RE   = re.compile(r"\[Merge-Stub\b|XSOURCE-MERGE")
_VER_RE    = re.compile(r"_(v[\d.]+(?:\.\d+)*)(?:_run\d+)?\.log$")
_KEY_RE    = re.compile(r"^([a-z]+)_")
_WORDS_RE  = re.compile(r"(\d[\d.]*)\s+W")
_PAGES_RE  = re.compile(r"(\d+)\s+S\.")
_CHUNKS_RE = re.compile(r"(\d+)\s+Chunks")

_PDF_LABELS: dict[str, str] = {
    "bates":    "Bates 2017",
    "kuhlthau": "Kuhlthau ISP",
    "schlebbe": "Schlebbe & Greifeneder 2022",
}

# Schwellenwerte gut/ok/schlecht — belegte Richtwerte
# Quellen: Vectara Hallucination Leaderboard Mai 2026, RAGAS-Benchmarks
#   Fehlerquote:   Claude Sonnet-4.6 = 10.6 % (Vectara); gut <10 %, ok 10–20 %, schlecht >20 %
#   Abdeckung:     RAGAS Context Recall Ziel >0.80; akademische Texte: gut ≥50 %, ok 30–50 %
#   Akzeptanzrate: Knowledge Extraction 45–70 % normal; gut ≥70 %, ok 50–70 %
THRESH_ACCEPT = (85, 65)   # gut ≥85 %, ok 65–85 %, schlecht <65 %
THRESH_HALL   = ( 5, 15)   # gut <5 %,  ok 5–15 %,  schlecht >15 %  — invert=True
THRESH_COV    = (80, 50)   # gut ≥80 %, ok 50–80 %, schlecht <50 %

# Claude Design Farbpalette (editorial, kein Neon)
_PDF_COLORS: dict[str, str] = {
    "bates":    "#e07a5f",  # coral
    "kuhlthau": "#5bbfbf",  # teal
    "schlebbe": "#e8b53b",  # amber
}
_COLOR_FALLBACKS = ["#8a86c8", "#6dbf8c", "#94a3b8"]

_PDF_META: dict[str, dict] = {
    "bates": {
        "titel":   "Information Behavior",
        "autor":   "Marcia J. Bates",
        "jahr":    "2017",
        "in":      "Encyclopedia of Library and Information Sciences, 3rd ed.",
        "thema":   "Grundlagentext des Felds Information Behavior. Definiert Kernbegriffe "
                   "(Information Seeking, Information Searching, Browsing), zeichnet die "
                   "Begriffsgeschichte von Use Studies bis Information Behavior nach und "
                   "stellt Bates' eigene Konzepte vor (Red Thread of Information, Berrypicking).",
        "sprache": "Englisch",
        "typ":     "Handbuchkapitel / Überblicksartikel",
    },
    "kuhlthau": {
        "titel":   "Information Search Process (ISP)",
        "autor":   "Carol C. Kuhlthau",
        "jahr":    "2009",
        "in":      "Eigenständiges Dokument / Buchkapitel",
        "thema":   "Beschreibt das ISP-Modell mit seinen 6 Phasen (Initiation, Selection, "
                   "Exploration, Formulation, Collection, Presentation). Jede Phase umfasst "
                   "drei Erfahrungsdimensionen: kognitiv, affektiv, physisch. "
                   "Zentrale Konzepte: Uncertainty Principle, Zone of Intervention.",
        "sprache": "Englisch",
        "typ":     "Theoriemodell-Dokument",
    },
    "schlebbe": {
        "titel":   "Information Need, Informationsbedarf und -bedürfnis",
        "autor":   "Kirsten Schlebbe & Elke Greifeneder",
        "jahr":    "2022",
        "in":      "Grundlagen der Informationswissenschaft (Kuhlen et al., Hrsg.)",
        "thema":   "Deutschsprachiger Überblick über Konzepte des Informationsbedarfs. "
                   "Behandelt Taylors Vier-Stufen-Typologie, Wilsons Modell (Information Needs "
                   "als sekundäre Bedürfnisse), Greens Vier-Charakteristiken und "
                   "Chatmans Small Worlds Theory.",
        "sprache": "Deutsch",
        "typ":     "Handbuchkapitel",
    },
}

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _ver_sort_key(v: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", v))

def _latest_version(ver_map: dict) -> str:
    return sorted(ver_map.keys(), key=_ver_sort_key)[-1]

def _median(lst: list[float]) -> float:
    s = sorted(lst)
    return s[len(s) // 2]

def _pdf_short_name(raw: str) -> str:
    name = raw.replace(".pdf", "").strip()
    parts = [p.strip() for p in name.split(" - ")]
    if len(parts) >= 2:
        return f"{parts[0]} ({parts[1]})" if parts[1].isdigit() else parts[0]
    name = re.sub(r"^\d+\.", "", name).strip()
    return name[:45]

def _pdf_color(key: str, idx: int = 0) -> str:
    return _PDF_COLORS.get(key, _COLOR_FALLBACKS[idx % len(_COLOR_FALLBACKS)])

# ---------------------------------------------------------------------------
# Daten lesen
# ---------------------------------------------------------------------------

def _read_quality_history() -> list[dict]:
    if not QUALITY_HISTORY.exists():
        return []
    rows = []
    for line in QUALITY_HISTORY.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _parse_log_extended(log: Path) -> dict:
    notes: dict[str, str] = {}
    words = pages = chunks = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _NOTE_RE.match(line)
        if m:
            routing, stem = m.group(1), m.group(2)
            if routing == "Vault" or "[Vault-Empf.]" in line:
                notes[stem] = "Vault"
            elif _MERGE_RE.search(line):
                notes[stem] = "Merge"
            else:
                notes[stem] = "Inbox"
        if words is None:
            mw = _WORDS_RE.search(line)
            if mw:
                try:
                    words = int(mw.group(1).replace(".", ""))
                except ValueError:
                    pass
        if pages is None:
            mp = _PAGES_RE.search(line)
            if mp:
                pages = int(mp.group(1))
        if chunks is None:
            mc = _CHUNKS_RE.search(line)
            if mc:
                chunks = int(mc.group(1))
    return {"notes": notes, "words": words, "pages": pages, "chunks": chunks}


def _log_key(log: Path) -> str | None:
    m = _KEY_RE.match(log.stem)
    return m.group(1) if m else None

def _log_version(log: Path) -> str | None:
    m = _VER_RE.search(log.name)
    return m.group(1) if m else None


def _read_all_log_runs() -> list[dict]:
    if not EVAL_DIR.exists():
        return []
    runs = []
    for log in sorted(EVAL_DIR.glob("*.log")):
        key = _log_key(log)
        if not key:
            continue
        ver   = _log_version(log) or "unknown"
        ext   = _parse_log_extended(log)
        notes = ext["notes"]
        if not notes:
            continue
        n_total = len(notes)
        n_vault = sum(1 for v in notes.values() if v == "Vault")
        n_merge = sum(1 for v in notes.values() if v == "Merge")
        n_inbox = sum(1 for v in notes.values() if v == "Inbox")
        runs.append({
            "key":        key,
            "label":      _PDF_LABELS.get(key, key),
            "ver":        ver,
            "n_total":    n_total,
            "n_vault":    n_vault,
            "n_merge":    n_merge,
            "n_inbox":    n_inbox,
            # Creation-Rate: neue Notes → Vault
            "accept_pct": round(100 * n_vault / n_total, 1) if n_total else 0.0,
            # Enrichment-Rate: Merge-Stubs (korrekte Ergänzung bestehender Notes)
            "enrich_pct": round(100 * n_merge / n_total, 1) if n_total else 0.0,
            # Erfolgsrate: Vault + Merge-Stubs zusammen
            "success_pct": round(100 * (n_vault + n_merge) / n_total, 1) if n_total else 0.0,
            "words":     ext["words"],
            "pages":     ext["pages"],
            "chunks":    ext["chunks"],
        })
    return runs


def _build_log_data(all_runs: list[dict]) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = {}
    for r in all_runs:
        data.setdefault(r["key"], {}).setdefault(r["ver"], []).append(r["accept_pct"])
    return data


def _read_token_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for jl in sorted(RUNS_DIR.glob("*.jsonl")):
        tin = tout = tcr = tcw = dur_ms = count = 0
        for line in jl.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("cached"):
                    continue
                tin    += r.get("input_tokens", 0) or 0
                tout   += r.get("output_tokens", 0) or 0
                tcr    += r.get("cache_read_tokens", 0) or 0
                tcw    += r.get("cache_creation_tokens", 0) or 0
                dur_ms += r.get("duration_ms", 0) or 0
                count  += 1
            except json.JSONDecodeError:
                pass
        if count > 0:
            # YYYYMMDD-HHMMSS → DD.MM HH:MM
            stem = jl.stem
            try:
                date_label = f"{stem[6:8]}.{stem[4:6]} {stem[9:11]}:{stem[11:13]}"
            except (IndexError, ValueError):
                date_label = stem
            runs.append({
                "date":         date_label,
                "run_id":       stem,
                "pdf_label":    "",
                "tokens_in":    tin,
                "tokens_out":   tout,
                "tokens_cache": tcr + tcw,
                "duration_min": round(dur_ms / 60000, 1),
                "calls":        count,
            })
    return runs

# ---------------------------------------------------------------------------
# Chart-Daten aufbereiten
# ---------------------------------------------------------------------------

def _calc_kpis(
    log_data:     dict[str, dict[str, list[float]]],
    all_log_runs: list[dict],
    quality_rows: list[dict],
    token_runs:   list[dict],
) -> dict:
    # KPIs = neueste Pipeline-Version, nicht Durchschnitt aller Versionen
    all_pvers = sorted({r.get("version") or r.get("pipeline_version") or "" for r in quality_rows if r.get("version") or r.get("pipeline_version")}, key=_ver_sort_key)
    latest_pver = all_pvers[-1] if all_pvers else None
    latest_qrows = [r for r in quality_rows if (r.get("version") or r.get("pipeline_version")) == latest_pver] if latest_pver else quality_rows

    accept_rates = [_median(vm[_latest_version(vm)]) for vm in log_data.values() if vm]
    avg_accept   = round(statistics.mean(accept_rates), 1) if accept_rates else None

    hall_rates = [r["hallucination_rate"] for r in latest_qrows
                  if "hallucination_rate" in r and r["hallucination_rate"] >= 0]
    avg_hall   = round(_median(hall_rates) * 100, 1) if hall_rates else None

    cov_vals = [v for r in latest_qrows
                if (v := r.get("coverage_factual") or r.get("coverage_rate")) is not None and v >= 0]
    avg_cov  = round(_median(cov_vals) * 100, 1) if cov_vals else None

    all_versions    = sorted({r["ver"] for r in all_log_runs}, key=_ver_sort_key)
    total_generated = sum(r["n_total"] for r in all_log_runs)
    total_accepted  = sum(r["n_vault"] for r in all_log_runs)
    total_merged    = sum(r.get("n_merge", 0) for r in all_log_runs)
    total_tokens    = sum(r["tokens_in"] + r["tokens_out"] for r in token_runs)
    total_dur_s     = sum(r["duration_min"] * 60 for r in token_runs)
    latest_truns    = [r for r in token_runs if r.get("ver") == latest_pver] if latest_pver else token_runs
    cur_tokens      = sum(r["tokens_in"] + r["tokens_out"] for r in latest_truns)
    cur_dur_h       = round(sum(r["duration_min"] for r in latest_truns) / 60, 1)
    cur_cost_usd    = round(sum(r.get("cost_usd", 0.0) or 0.0 for r in latest_truns), 4)

    return {
        "avg_accept":      avg_accept,
        "avg_hall":        avg_hall,
        "avg_cov":         avg_cov,
        "kpi_version":     latest_pver,
        "n_notes":         len(latest_qrows),
        "total_runs":      len(all_log_runs),
        "n_pdfs":          len(log_data),
        "n_versions":      len(all_versions),
        "versions_range":  f"{all_versions[0]}–{all_versions[-1]}" if len(all_versions) > 1 else (all_versions[0] if all_versions else "--"),
        "total_generated": total_generated,
        "total_accepted":  total_accepted,
        "total_merged":    total_merged,
        "total_tokens":    total_tokens,
        "total_dur_h":     round(total_dur_s / 3600, 1),
        "cur_tokens":      cur_tokens,
        "cur_dur_h":       cur_dur_h,
        "cur_cost_usd":    cur_cost_usd,
    }


def _calc_pdf_table(
    log_data:     dict[str, dict[str, list[float]]],
    all_log_runs: list[dict],
    quality_rows: list[dict],
) -> list[dict]:
    rows = []
    for key in sorted(log_data):
        ver_map = log_data[key]
        if not ver_map:
            continue
        latest    = _latest_version(ver_map)
        accept    = _median(ver_map[latest])
        label     = _PDF_LABELS.get(key, key)
        pdf_runs  = [r for r in all_log_runs if r["key"] == key]
        words_list = [r["words"] for r in pdf_runs if r["words"]]
        pages_list = [r["pages"] for r in pdf_runs if r["pages"]]
        words = _median(words_list) if words_list else None
        pages = _median(pages_list) if pages_list else None
        pdf_qrows = [r for r in quality_rows if key.lower() in (r.get("pdf") or "").lower()]
        _hall_vals = [r["hallucination_rate"] for r in pdf_qrows
                      if r.get("hallucination_rate") is not None and r["hallucination_rate"] >= 0]
        hall = round(statistics.mean(_hall_vals) * 100, 1) if _hall_vals else None
        _cov_vals = [v for r in pdf_qrows
                     if (v := r.get("coverage_factual") or r.get("coverage_rate", -1)) >= 0]
        cov  = round(statistics.mean(_cov_vals) * 100, 1) if _cov_vals else None
        rows.append({
            "key":     key,
            "label":   label,
            "version": latest,
            "accept":  accept,
            "hall":    hall,
            "cov":     cov,
            "n_notes": len(pdf_qrows),
            "words":   int(words) if words else None,
            "pages":   int(pages) if pages else None,
        })
    return rows


def _chart_acceptance(log_data: dict) -> dict:
    labels, values, colors = [], [], []
    for key in sorted(log_data):
        vm = log_data[key]
        if not vm:
            continue
        labels.append(_PDF_LABELS.get(key, key))
        values.append(_median(vm[_latest_version(vm)]))
        colors.append(_pdf_color(key))
    return {"labels": labels, "values": values, "colors": colors}


def _chart_scatter(quality_rows: list[dict]) -> dict:
    points: list[dict] = []
    pdf_map: dict[str, str] = {}
    for r in quality_rows:
        hall = r.get("hallucination_rate")
        cov  = r.get("coverage_factual") or r.get("coverage_rate")
        if hall is None or cov is None or float(hall) < 0 or float(cov) < 0:
            continue
        label = r.get("note") or r.get("note_title") or "?"
        label = re.sub(r"^(vault|inbox)__", "", label).replace(".md", "")
        pdf   = r.get("pdf") or r.get("source_pdf") or "unbekannt"
        if pdf not in pdf_map:
            pdf_map[pdf] = _pdf_short_name(pdf)
        points.append({
            "x": round(float(hall) * 100, 1),
            "y": round(float(cov)  * 100, 1),
            "label": label,
            "pdf":   pdf,
            "pdf_label": pdf_map[pdf],
        })
    pdfs = [{"raw": k, "label": v} for k, v in pdf_map.items()]
    return {"points": points, "pdfs": pdfs}


def _chart_longitudinal(log_data: dict) -> dict:
    all_ver: set[str] = set()
    for vm in log_data.values():
        all_ver.update(vm.keys())
    versions = sorted(all_ver, key=_ver_sort_key)
    datasets = []
    for key in sorted(log_data):
        vm       = log_data[key]
        data_pts = [_median(vm[v]) if vm.get(v) else None for v in versions]
        datasets.append({
            "label": _PDF_LABELS.get(key, key),
            "data":  data_pts,
            "color": _pdf_color(key),
        })
    return {"versions": versions, "datasets": datasets}


def _chart_tokens(runs: list[dict]) -> dict:
    return {
        "labels":       [r["date"]         for r in runs],
        "pdf_labels":   [r.get("pdf_label", "") for r in runs],
        "tokens_in":    [r["tokens_in"]    for r in runs],
        "tokens_out":   [r["tokens_out"]   for r in runs],
        "tokens_cache": [r["tokens_cache"] for r in runs],
        "duration_min": [r["duration_min"] for r in runs],
    }


def _chart_scaling(all_log_runs: list[dict]) -> dict:
    points = [
        {
            "x":       r["words"],
            "y":       r["n_total"],
            "y_vault": r["n_vault"],
            "pages":   r["pages"],
            "key":     r["key"],
            "label":   r["label"],
            "ver":     r["ver"],
            "pct":     r["accept_pct"],
        }
        for r in all_log_runs
        if r["words"] is not None
    ]
    keys = sorted({p["key"] for p in points})
    return {"points": points, "keys": keys}


def _build_quality_chart_data(quality_rows: list[dict]) -> dict:
    """Bereitet quality_history-Daten fuer Client-seitige Charts und Filter auf."""
    rows_clean = []
    for r in quality_rows:
        note = r.get("note") or r.get("note_title") or "?"
        note = re.sub(r"^(vault|inbox)__", "", note).replace(".md", "")
        pdf  = r.get("pdf") or r.get("source_pdf") or "unbekannt"
        ver  = r.get("version") or "unknown"
        hall = r.get("hallucination_rate")
        cov  = r.get("coverage_factual") or r.get("coverage_rate")
        anch_total = r.get("anchors_total") or 0
        anch_conf  = r.get("anchors_confirmed") or 0
        rows_clean.append({
            "note":     note,
            "pdf":      pdf,
            "pdf_short": _pdf_short_name(pdf),
            "version":  ver,
            "hall":     round(float(hall) * 100, 1) if hall is not None else None,
            "cov":      round(float(cov)  * 100, 1) if cov  is not None else None,
            "anchors_confirmed": anch_conf,
            "anchors_total":     anch_total,
            "tokens_input":  r.get("tokens_input", 0) or 0,
            "tokens_output": r.get("tokens_output", 0) or 0,
            "tokens_cache":  r.get("tokens_cache_read", 0) or 0,
            "wall_time_s":   r.get("wall_time_s", 0) or 0,
            "small_sample":  r.get("small_sample_warning", False),
        })

    all_versions = sorted({r["version"] for r in rows_clean}, key=_ver_sort_key)
    all_pdfs     = sorted({r["pdf"] for r in rows_clean})

    # Slope-Daten: Median Hall-Rate + Coverage pro Version pro PDF
    slope_datasets = []
    pdf_colors_used: dict[str, str] = {}
    for i, pdf in enumerate(all_pdfs):
        key = _KEY_RE.match(pdf.replace(".pdf","").strip()).group(1) if _KEY_RE.match(pdf.replace(".pdf","").strip()) else pdf.lower()[:6]
        color = _pdf_color(key, i)
        pdf_colors_used[pdf] = color
        hall_pts = []
        cov_pts  = []
        for v in all_versions:
            vrows = [r for r in rows_clean if r["pdf"] == pdf and r["version"] == v and r["hall"] is not None]
            hall_pts.append(_median([r["hall"] for r in vrows]) if vrows else None)
            vrows2 = [r for r in rows_clean if r["pdf"] == pdf and r["version"] == v and r["cov"] is not None]
            cov_pts.append(_median([r["cov"] for r in vrows2]) if vrows2 else None)
        slope_datasets.append({
            "pdf":       pdf,
            "pdf_short": _pdf_short_name(pdf),
            "color":     color,
            "hall_data": hall_pts,
            "cov_data":  cov_pts,
        })

    # Token-Daten: Summe Input/Output/Cache pro Version
    token_by_ver: dict[str, dict] = {}
    for v in all_versions:
        vrows = [r for r in rows_clean if r["version"] == v]
        token_by_ver[v] = {
            "tokens_input":  sum(r["tokens_input"]  for r in vrows),
            "tokens_output": sum(r["tokens_output"] for r in vrows),
            "tokens_cache":  sum(r["tokens_cache"]  for r in vrows),
            "n":             len(vrows),
        }

    return {
        "rows":           rows_clean,
        "versions":       all_versions,
        "pdfs":           all_pdfs,
        "pdf_colors":     pdf_colors_used,
        "slope_datasets": slope_datasets,
        "token_by_ver":   token_by_ver,
    }

# ---------------------------------------------------------------------------
# HTML-Render-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _pill(value, good_thr, bad_thr, invert=False, suffix="%", good_label=None, bad_label=None) -> str:
    """Gibt eine .pill.good/warn/bad Spanne zurueck."""
    if value is None:
        return '<span class="pill flat">--</span>'
    if invert:
        cls = "good" if value <= good_thr else ("bad" if value >= bad_thr else "warn")
    else:
        cls = "good" if value >= good_thr else ("bad" if value <= bad_thr else "warn")
    if cls == "good" and good_label:
        text = good_label
    elif cls == "bad" and bad_label:
        text = bad_label
    else:
        text = f"{value}{suffix}"
    return f'<span class="pill {cls}">{text}</span>'


def _quant(value, good_thr, bad_thr, invert=False, suffix="%") -> str:
    """Gibt einen .quant.good/warn/bad Span zurueck."""
    if value is None:
        return '<span class="dash">&mdash;</span>'
    if invert:
        cls = "good" if value <= good_thr else ("bad" if value >= bad_thr else "warn")
    else:
        cls = "good" if value >= good_thr else ("bad" if value <= bad_thr else "warn")
    val_str = f"{value:,.1f}".replace(",", ".") if isinstance(value, float) else str(value)
    return f'<span class="quant {cls}">{val_str}{suffix}</span>'


def _mini_bar(value, good_thr, bad_thr) -> str:
    """Mini-Balken Für Akzeptanzrate in Tabelle."""
    if value is None:
        return ""
    cls = "good" if value >= good_thr else ("bad" if value <= bad_thr else "warn")
    return f'<span class="mini-bar {cls}"><i style="width:{min(value,100):.0f}%"></i></span>'


def _render_pdf_table(rows: list[dict]) -> str:
    if not rows:
        return '<p style="color:var(--ink-4);font-style:italic;padding:2rem 0">Keine Daten vorhanden.</p>'

    header = (
        '<table class="cmp"><thead><tr>'
        '<th style="width:28px"></th>'
        '<th>Quell-PDF</th>'
        '<th>Letzte Version</th>'
        '<th class="num">W&ouml;rter</th>'
        '<th class="num">Seiten</th>'
        '<th class="num">Akzeptiert</th>'
        '<th class="num">Fehlerquote</th>'
        '<th class="num">Abdeckung</th>'
        '<th class="num">Eval.&nbsp;Notes</th>'
        '</tr></thead><tbody>'
    )
    body = ""
    for r in rows:
        key    = r.get("key", "")
        meta   = _PDF_META.get(key, {})
        w_str  = f"{r['words']:,}".replace(",", ".") if r["words"] else "&mdash;"
        p_str  = str(r["pages"]) if r["pages"] else "&mdash;"
        acc_q  = _quant(r["accept"], THRESH_ACCEPT[0], THRESH_ACCEPT[1])
        hall_q = _quant(r["hall"], THRESH_HALL[0], THRESH_HALL[1], invert=True)
        cov_q  = _quant(r["cov"], THRESH_COV[0], THRESH_COV[1])
        bar    = _mini_bar(r["accept"], THRESH_ACCEPT[0], THRESH_ACCEPT[1])
        ver_cls = "cur" if r["version"] != "unknown" else ""

        if meta:
            def _dl(k: str, v: str) -> str:
                return f'<div class="pdf-dl-row"><dt>{k}</dt><dd>{v}</dd></div>'
            meta_html = (
                '<dl class="pdf-meta">'
                + _dl("Vollst. Titel", meta.get("titel", "--"))
                + _dl("Autor(en)", meta.get("autor", "--"))
                + _dl("Jahr", meta.get("jahr", "--"))
                + _dl("Erschienen in", meta.get("in", "--"))
                + _dl("Sprache / Typ", f'{meta.get("sprache","--")} &middot; {meta.get("typ","--")}')
                + f'<div class="pdf-dl-row pdf-dl-full"><dt>Inhalt</dt><dd>{meta.get("thema","--")}</dd></div>'
                + '</dl>'
            )
            toggle = (
                '<td style="text-align:center;width:28px">'
                '<button class="expand-btn" onclick="toggleRow(this)" title="Details">&#9656;</button>'
                '</td>'
            )
            detail = (
                f'<tr class="detail-row" style="display:none">'
                f'<td colspan="9" style="padding:0;border-bottom:1px solid var(--hairline)">'
                f'{meta_html}</td></tr>'
            )
        else:
            toggle = '<td></td>'
            detail = ''

        body += (
            f'<tr class="data-row">'
            f'{toggle}'
            f'<td class="td-name" onclick="toggleRow(this.closest(\'tr\').querySelector(\'.expand-btn\'))" style="cursor:pointer">{r["label"]}</td>'
            f'<td><span class="tag {ver_cls}">{r["version"]}</span></td>'
            f'<td class="num" style="color:var(--ink-3)">{w_str}</td>'
            f'<td class="num" style="color:var(--ink-3)">{p_str}</td>'
            f'<td class="num">{bar}{acc_q}</td>'
            f'<td class="num">{hall_q}</td>'
            f'<td class="num">{cov_q}</td>'
            f'<td class="num" style="color:var(--ink-3)">{r["n_notes"] or "&mdash;"}</td>'
            f'</tr>'
            f'{detail}'
        )
    return header + body + "</tbody></table>"

# ---------------------------------------------------------------------------
# HTML zusammenbauen
# ---------------------------------------------------------------------------

_CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
_FONT_URL    = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Inter+Tight:wght@500;600;700;800&display=swap"


def _build_html(
    kpis:           dict,
    pdf_table_rows: list[dict],
    accept_chart:   dict,
    scatter_chart:  dict,
    long_chart:     dict,
    token_chart:    dict,
    scaling_chart:  dict,
    quality_data:   dict,
    generated_at:   str,
) -> str:
    accept_json   = json.dumps(accept_chart,  ensure_ascii=False)
    scatter_json  = json.dumps(scatter_chart, ensure_ascii=False)
    long_json     = json.dumps(long_chart,    ensure_ascii=False)
    token_json    = json.dumps(token_chart,   ensure_ascii=False)
    scaling_json  = json.dumps(scaling_chart, ensure_ascii=False)
    quality_json  = json.dumps(quality_data,  ensure_ascii=False)

    accept_empty  = not accept_chart.get("labels")
    scatter_empty = not scatter_chart.get("points")
    long_empty    = not long_chart.get("versions")
    token_empty   = not token_chart.get("labels")
    scaling_empty = not scaling_chart.get("points")
    quality_empty = not quality_data.get("rows")

    # Scatter-Filterleiste (Chart 2)
    pdf_filter_html = ""
    if not scatter_empty and scatter_chart.get("pdfs"):
        btns = '<button class="filter-btn active" data-pdf="__all__">Alle PDFs</button>'
        for p in scatter_chart["pdfs"]:
            btns += f'<button class="filter-btn" data-pdf="{p["raw"]}">{p["label"]}</button>'
        pdf_filter_html = f'<div class="filter-bar" id="scatterFilter">{btns}</div>'

    tok_m = f'{kpis["total_tokens"] / 1_000_000:.2f}M' if kpis["total_tokens"] else "--"

    pdf_table_html = _render_pdf_table(pdf_table_rows)

    no_data = '<p style="color:var(--ink-4);font-style:italic;padding:3rem 0;text-align:center">Keine Daten vorhanden.</p>'

    # Versions- und PDF-Optionen fuer Filter-Dropdowns
    all_versions_opts = "".join(
        f'<option value="{v}">{v}</option>'
        for v in quality_data.get("versions", [])
    )
    all_pdfs_opts = "".join(
        f'<option value="{p}">{_pdf_short_name(p)}</option>'
        for p in quality_data.get("pdfs", [])
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Atomic Agent &mdash; Eval Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_FONT_URL}" rel="stylesheet">
<script src="{_CHARTJS_CDN}"></script>
<style>
:root {{
  --bg:         #0f172a;
  --bg-elev:    #131e36;
  --bg-soft:    rgba(255,255,255,0.018);
  --hairline:   rgba(148,163,184,0.14);
  --hairline-2: rgba(148,163,184,0.22);
  --ink-1: #f1f5f9;
  --ink-2: #cbd5e1;
  --ink-3: #94a3b8;
  --ink-4: #64748b;
  --ink-5: #475569;
  --c-amber:  #e8b53b;
  --c-teal:   #5bbfbf;
  --c-coral:  #e07a5f;
  --c-violet: #8a86c8;
  --c-mint:   #6dbf8c;
  --good: #6dbf8c;
  --warn: #e8b53b;
  --bad:  #e07a5f;
  --maxw: 1280px;
  --pad-x: clamp(20px, 4vw, 64px);
}}
*,*::before,*::after {{ box-sizing: border-box; }}
html, body {{ margin: 0; }}
body {{
  background: var(--bg);
  color: var(--ink-2);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  font-feature-settings: 'ss01','cv05','cv11';
  padding-bottom: 96px;
}}
body::before {{
  content: "";
  position: fixed; inset: 0;
  pointer-events: none;
  background-image: linear-gradient(rgba(148,163,184,0.025) 1px, transparent 1px);
  background-size: 100% 96px;
  z-index: 0;
  mask-image: linear-gradient(180deg, transparent, black 200px, black calc(100% - 200px), transparent);
}}
.wrap {{
  max-width: var(--maxw);
  margin: 0 auto;
  padding: 56px var(--pad-x) 0;
  position: relative; z-index: 1;
}}
/* Eyebrow */
.eyebrow {{
  font-size: 10.5px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.18em; color: var(--ink-4);
}}
.eyebrow .dot {{
  display: inline-block; width: 6px; height: 6px;
  background: var(--c-amber); border-radius: 50%;
  vertical-align: middle; margin-right: 10px; transform: translateY(-1px);
}}
/* Header */
.hdr {{
  display: grid; grid-template-columns: 1fr auto;
  gap: 48px; align-items: end;
  padding-bottom: 28px; border-bottom: 1px solid var(--hairline);
}}
.hdr h1 {{
  font-family: 'Inter Tight', 'Inter', sans-serif;
  font-size: clamp(28px, 3.8vw, 46px);
  font-weight: 700; letter-spacing: -0.025em; line-height: 1.05;
  color: var(--ink-1); margin: 14px 0 16px;
}}
.hdr .lead {{ max-width: 62ch; font-size: 16px; line-height: 1.6; color: var(--ink-2); margin: 0; }}
.hdr .lead em {{
  font-style: normal; color: var(--ink-1);
  background: linear-gradient(transparent 62%, rgba(232,181,59,0.22) 62%);
  padding: 0 2px;
}}
.hdr .meta {{ text-align: right; color: var(--ink-4); font-size: 12px; line-height: 1.7; font-variant-numeric: tabular-nums; }}
.hdr .meta b {{ color: var(--ink-2); font-weight: 500; }}
.hdr .meta code {{
  font-family: 'Inter', monospace; color: var(--ink-3);
  background: var(--bg-soft); padding: 1px 5px;
  border: 1px solid var(--hairline); border-radius: 3px; font-size: 11px;
}}
/* Strip */
.strip {{
  display: grid; grid-template-columns: repeat(7, 1fr);
  gap: 0; border-bottom: 1px solid var(--hairline);
  padding: 22px 0 26px;
}}
.strip .cell {{ padding: 0 18px; border-left: 1px solid var(--hairline); }}
.strip .cell:first-child {{ border-left: 0; padding-left: 0; }}
.strip .k {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em; color: var(--ink-4); font-weight: 500; margin-bottom: 6px; }}
.strip .v {{ font-family: 'Inter Tight', sans-serif; font-size: 18px; font-weight: 600; color: var(--ink-1); font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
.strip .v small {{ color: var(--ink-4); font-weight: 400; font-size: 12px; margin-left: 2px; }}
/* Section head */
.section-head {{ margin: 48px 0 22px; display: flex; align-items: baseline; gap: 18px; }}
.section-head h2 {{ font-family: 'Inter Tight', sans-serif; font-size: 21px; font-weight: 700; letter-spacing: -0.015em; color: var(--ink-1); margin: 0; }}
.section-head .rule {{ flex: 1; height: 1px; background: var(--hairline); }}
.section-head .note {{ font-size: 12px; color: var(--ink-4); white-space: nowrap; }}
/* KPIs */
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; border-top: 1px solid var(--hairline); border-bottom: 1px solid var(--hairline); }}
.kpi {{ padding: 24px 28px 26px; border-left: 1px solid var(--hairline); position: relative; }}
.kpi:first-child {{ border-left: 0; padding-left: 4px; }}
.kpi .k {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.16em; font-weight: 600; color: var(--ink-4); margin-bottom: 14px; }}
.kpi .v {{ font-family: 'Inter Tight', sans-serif; font-size: 52px; font-weight: 700; color: var(--ink-1); line-height: 0.95; letter-spacing: -0.035em; font-variant-numeric: tabular-nums; }}
.kpi .v .unit {{ font-size: 24px; font-weight: 500; color: var(--ink-3); margin-left: 2px; letter-spacing: -0.01em; }}
.kpi .delta {{ margin-top: 14px; display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--ink-3); flex-wrap: wrap; }}
.kpi .desc {{ margin-top: 6px; font-size: 12.5px; color: var(--ink-4); max-width: 28ch; line-height: 1.45; }}
/* Pills */
.pill {{
  display: inline-block; font-size: 10px; font-weight: 600;
  letter-spacing: 0.06em; padding: 2px 8px; border-radius: 3px;
  text-transform: uppercase; white-space: nowrap;
}}
.pill.good {{ background: rgba(109,191,140,0.13); color: var(--good); }}
.pill.warn {{ background: rgba(232,181,59,0.13);  color: var(--warn); }}
.pill.bad  {{ background: rgba(224,122,95,0.14);  color: var(--bad);  }}
.pill.flat {{ background: rgba(148,163,184,0.12); color: var(--ink-3); }}
/* Table */
.table-wrap {{ margin-top: 4px; overflow-x: auto; }}
table.cmp {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
table.cmp thead th {{
  text-align: left; font-size: 10px; font-weight: 600;
  letter-spacing: 0.15em; text-transform: uppercase; color: var(--ink-4);
  padding: 14px 12px; border-bottom: 1px solid var(--hairline-2);
}}
table.cmp thead th:first-child {{ padding-left: 0; }}
table.cmp thead th.num {{ text-align: right; }}
table.cmp tbody td {{
  padding: 16px 12px; border-bottom: 1px solid var(--hairline);
  color: var(--ink-2); font-size: 14px;
}}
table.cmp tbody td:first-child {{ padding-left: 0; }}
table.cmp tbody tr:last-child td {{ border-bottom: 0; }}
table.cmp tbody tr.data-row:hover td {{ background: rgba(255,255,255,0.018); }}
table.cmp td.num {{ text-align: right; }}
.tag {{
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; padding: 2px 8px 3px; border-radius: 3px;
  color: var(--ink-3); background: rgba(148,163,184,0.10); font-weight: 500;
}}
.tag.cur {{ color: var(--ink-1); background: rgba(232,181,59,0.13); }}
.quant {{ font-weight: 600; color: var(--ink-1); }}
.quant.good {{ color: var(--good); }}
.quant.warn {{ color: var(--warn); }}
.quant.bad  {{ color: var(--bad);  }}
.dash {{ color: var(--ink-5); }}
.mini-bar {{
  display: inline-block; width: 48px; height: 5px;
  background: rgba(148,163,184,0.10); border-radius: 1px;
  margin-right: 8px; vertical-align: middle;
  overflow: hidden; position: relative; top: -1px;
}}
.mini-bar > i {{ display: block; height: 100%; background: var(--ink-4); }}
.mini-bar.good > i {{ background: var(--good); }}
.mini-bar.warn > i {{ background: var(--warn); }}
.mini-bar.bad  > i {{ background: var(--bad);  }}
/* Expand-Button */
.expand-btn {{
  background: none; border: none; cursor: pointer; color: var(--ink-5);
  font-size: 10px; padding: 3px 5px; border-radius: 3px;
  transition: color 0.15s, background 0.15s; line-height: 1;
}}
.expand-btn:hover {{ color: var(--c-amber); background: rgba(232,181,59,0.08); }}
.expand-btn.open  {{ color: var(--c-amber); display: inline-block; transform: rotate(90deg); }}
/* PDF-Metadaten */
.pdf-meta {{
  margin: 0; padding: 1rem 1.4rem 1.1rem 2.5rem;
  background: rgba(15,23,42,0.55);
  display: grid; grid-template-columns: 1fr 1fr; gap: 0 3rem;
}}
.pdf-dl-row {{
  display: grid; grid-template-columns: 130px 1fr;
  gap: 0 0.6rem; align-items: baseline;
  padding: 0.28rem 0; border-bottom: 1px solid var(--hairline);
}}
.pdf-dl-row:last-child {{ border-bottom: none; }}
.pdf-dl-full {{ grid-column: 1 / -1; grid-template-columns: 130px 1fr; }}
.pdf-meta dt {{ color: var(--ink-5); font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; }}
.pdf-meta dd {{ margin: 0; color: var(--ink-2); font-size: 13px; line-height: 1.5; }}
/* Charts */
.charts {{
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 1px; background: var(--hairline);
  border: 1px solid var(--hairline); margin-top: 4px;
}}
.chart {{ background: var(--bg); padding: 24px 26px 22px; min-width: 0; }}
.chart.wide {{ grid-column: 1 / -1; }}
.chart .head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 4px; }}
.chart h3 {{ font-family: 'Inter Tight', sans-serif; font-size: 15px; font-weight: 700; color: var(--ink-1); margin: 0; letter-spacing: -0.005em; }}
.chart .ax {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--ink-5); font-weight: 500; white-space: nowrap; flex-shrink: 0; }}
.chart .sub {{ font-size: 13px; color: var(--ink-3); max-width: 64ch; margin: 4px 0 14px; line-height: 1.5; }}
.chart .sub b {{ color: var(--ink-1); font-weight: 600; }}
.chart .body {{ position: relative; height: 280px; }}
.chart.wide .body {{ height: 300px; }}
/* Legend */
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--ink-3); margin-top: 10px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.legend i {{ display: inline-block; width: 8px; height: 8px; border-radius: 1px; }}
.legend i.round {{ border-radius: 50%; }}
/* Filter */
.filter-bar {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 12px; }}
.filter-btn {{
  background: rgba(30,58,95,0.6); border: 1px solid var(--hairline-2);
  color: var(--ink-4); border-radius: 3px; padding: 3px 10px;
  font-size: 11px; font-family: inherit; cursor: pointer; font-weight: 500;
  transition: background 0.15s, color 0.15s;
}}
.filter-btn:hover  {{ background: rgba(232,181,59,0.12); color: var(--c-amber); border-color: rgba(232,181,59,0.3); }}
.filter-btn.active {{ background: rgba(232,181,59,0.13); color: var(--c-amber); border-color: rgba(232,181,59,0.4); }}
/* Global sticky filter bar */
.global-filter-bar {{
  position: sticky; top: 0; z-index: 100;
  background: rgba(15,23,42,0.92); backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--hairline-2);
  padding: 10px var(--pad-x);
  margin: 0 calc(-1 * var(--pad-x));
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
}}
.global-filter-bar label {{
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em;
  color: var(--ink-4); font-weight: 600; white-space: nowrap;
}}
.global-filter-bar select {{
  background: rgba(30,58,95,0.7); border: 1px solid var(--hairline-2);
  color: var(--ink-2); border-radius: 3px; padding: 4px 10px;
  font-size: 12px; font-family: inherit; cursor: pointer;
  appearance: none; -webkit-appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2364748b'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 8px center;
  padding-right: 24px; min-width: 130px; max-width: 220px;
}}
.global-filter-bar select:focus {{ outline: none; border-color: rgba(232,181,59,0.5); }}
.global-filter-bar .reset-btn {{
  background: rgba(224,122,95,0.10); border: 1px solid rgba(224,122,95,0.25);
  color: var(--c-coral); border-radius: 3px; padding: 4px 12px;
  font-size: 11px; font-family: inherit; cursor: pointer; font-weight: 600;
  transition: background 0.15s;
}}
.global-filter-bar .reset-btn:hover {{ background: rgba(224,122,95,0.2); }}
.global-filter-bar .filter-count {{
  margin-left: auto; font-size: 11px; color: var(--ink-4);
  font-variant-numeric: tabular-nums;
}}
/* KPI cards (neue filternde Karten) */
.kpi-cards {{
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 0; border: 1px solid var(--hairline); margin-top: 24px;
}}
.kpi-card {{
  padding: 20px 24px 22px; border-left: 1px solid var(--hairline);
  position: relative;
}}
.kpi-card:first-child {{ border-left: 0; }}
.kpi-card .k {{
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.16em;
  font-weight: 600; color: var(--ink-4); margin-bottom: 10px;
}}
.kpi-card .v {{
  font-family: 'Inter Tight', sans-serif; font-size: 42px; font-weight: 700;
  color: var(--ink-1); line-height: 0.95; letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums;
}}
.kpi-card .v .unit {{ font-size: 20px; font-weight: 500; color: var(--ink-3); margin-left: 2px; }}
.kpi-card .v.good {{ color: var(--good); }}
.kpi-card .v.warn {{ color: var(--warn); }}
.kpi-card .v.bad  {{ color: var(--bad);  }}
/* Detail-Tabelle */
table.detail-tbl {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:13px; }}
table.detail-tbl thead th {{
  text-align:left; font-size:10px; font-weight:600;
  letter-spacing:0.14em; text-transform:uppercase; color:var(--ink-4);
  padding:12px 10px; border-bottom:1px solid var(--hairline-2); cursor:pointer;
  user-select:none; white-space:nowrap;
}}
table.detail-tbl thead th:hover {{ color:var(--c-amber); }}
table.detail-tbl thead th.num {{ text-align:right; }}
table.detail-tbl thead th .sort-icon {{ margin-left:4px; opacity:0.4; }}
table.detail-tbl thead th.sort-asc .sort-icon::after  {{ content:'▲'; opacity:1; }}
table.detail-tbl thead th.sort-desc .sort-icon::after {{ content:'▼'; opacity:1; }}
table.detail-tbl tbody td {{
  padding:11px 10px; border-bottom:1px solid var(--hairline);
  color:var(--ink-2);
}}
table.detail-tbl tbody tr:hover td {{ background:rgba(255,255,255,0.018); }}
table.detail-tbl tbody td.num {{ text-align:right; }}
table.detail-tbl tbody td.note-name {{ max-width:340px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--ink-1); font-size:12px; }}
/* Footer */
.foot {{
  margin-top: 56px; padding-top: 22px; border-top: 1px solid var(--hairline);
  color: var(--ink-4); font-size: 11.5px;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
}}
.foot code {{ font-family: 'Inter', monospace; color: var(--ink-3); }}
/* Responsive */
@media (max-width: 1000px) {{
  .charts {{ grid-template-columns: 1fr; }}
  .hdr {{ grid-template-columns: 1fr; gap: 16px; }}
  .hdr .meta {{ text-align: left; }}
  .strip {{ grid-template-columns: repeat(3, 1fr); row-gap: 16px; }}
  .strip .cell:nth-child(3n+1) {{ border-left: 0; padding-left: 0; }}
  .kpis {{ grid-template-columns: 1fr 1fr; }}
  .kpi:nth-child(odd) {{ border-left: 0; padding-left: 4px; }}
  .kpi:nth-child(n+3) {{ border-top: 1px solid var(--hairline); }}
}}
</style>
</head>
<body>
<div class="wrap">

<!-- Header -->
<header class="hdr">
  <div>
    <div class="eyebrow"><span class="dot"></span>Atomic Agent &middot; Pipeline Eval &middot; {generated_at}</div>
    <h1>Wie zuverlässig liest die Pipeline ein Paper?</h1>
    <p class="lead">
      Eine Notiz-Pipeline verarbeitet wissenschaftliche PDFs und entscheidet
      autonom, welche Wissensnotizen direkt ins System wandern &mdash; und welche
      zuerst <em>von Hand geprüft</em> werden muessen. Diese Auswertung zeigt,
      wo das Verfahren stabil läuft und wo noch Spielraum ist.
    </p>
  </div>
  <div class="meta">
    <b>Stand</b> {generated_at}<br>
    <b>Quellen</b> <code>quality_history.jsonl</code><br>
    <code>eval/baseline/*.log</code> &middot; <code>runs/*.jsonl</code>
  </div>
</header>

<!-- Datenbasis-Strip -->
<div class="strip" aria-label="Datenbasis">
  <div class="cell"><div class="k">PDFs</div><div class="v">{kpis["n_pdfs"]}</div></div>
  <div class="cell"><div class="k">Pipeline-Versionen</div><div class="v">{kpis["n_versions"]}<small> &middot; {kpis["versions_range"]}</small></div></div>
  <div class="cell"><div class="k">Runs</div><div class="v">{kpis["total_runs"]}</div></div>
  <div class="cell"><div class="k">Notes generiert</div><div class="v">{kpis["total_generated"]}</div></div>
  <div class="cell"><div class="k">Akzeptiert</div><div class="v">{kpis["total_accepted"]}<small> / {round(100*kpis["total_accepted"]/kpis["total_generated"],1) if kpis["total_generated"] else "--"}&nbsp;%</small></div></div>
  <div class="cell"><div class="k">Tokens</div><div class="v">{tok_m}</div></div>
  <div class="cell"><div class="k">Laufzeit</div><div class="v">{kpis["total_dur_h"]}<small>&nbsp;h</small></div></div>
</div>

<!-- ===== FILTER-BAR (sticky) ===== -->
<div class="global-filter-bar" id="globalFilter">
  <label>Version</label>
  <select id="filterVersion">
    <option value="">Alle</option>
    {all_versions_opts}
  </select>
  <label>PDF</label>
  <select id="filterPdf">
    <option value="">Alle</option>
    {all_pdfs_opts}
  </select>
  <button class="reset-btn" id="filterReset">Reset</button>
  <span class="filter-count" id="filterCount"></span>
</div>

<!-- ===== KPI-KARTEN (dynamisch) ===== -->
<div class="section-head" style="margin-top:28px">
  <h2>Kennzahlen</h2>
  <div class="rule"></div>
  <div class="note" id="kpiNote">gefiltertes Set</div>
</div>
<div class="kpi-cards">
  <div class="kpi-card">
    <div class="k">Halluzinationsrate &Oslash;</div>
    <div class="v" id="kpiHall">--<span class="unit">%</span></div>
  </div>
  <div class="kpi-card">
    <div class="k">Coverage &Oslash;</div>
    <div class="v" id="kpiCov">--<span class="unit">%</span></div>
  </div>
  <div class="kpi-card">
    <div class="k">N (evaluierte Notes)</div>
    <div class="v" id="kpiN">--</div>
  </div>
  <div class="kpi-card">
    <div class="k">Anchors best&auml;tigt</div>
    <div class="v" id="kpiAnchors">--<span class="unit">%</span></div>
  </div>
</div>

<!-- ===== SLOPE-CHART ===== -->
<div class="section-head">
  <h2>Hal-Rate &amp; Coverage &uuml;ber Versionen</h2>
  <div class="rule"></div>
  <div class="note">Median pro Version und PDF</div>
</div>
<div class="charts">
  <div class="chart">
    <div class="head"><h3>Halluzinationsrate je Version</h3><div class="ax">Median %</div></div>
    <p class="sub">Niedrig = besser. Jede Linie ein PDF.</p>
    {"<div class='body'><canvas id='chSlope1'></canvas></div><div class='legend' id='legSlope1'></div>" if not quality_empty else no_data}
  </div>
  <div class="chart">
    <div class="head"><h3>Coverage je Version</h3><div class="ax">Median %</div></div>
    <p class="sub">Hoch = besser. Jede Linie ein PDF.</p>
    {"<div class='body'><canvas id='chSlope2'></canvas></div><div class='legend' id='legSlope2'></div>" if not quality_empty else no_data}
  </div>
</div>

<!-- ===== TOKEN-CHART ===== -->
<div class="section-head">
  <h2>Token-Verbrauch pro Pipeline-Version</h2>
  <div class="rule"></div>
  <div class="note">Summe aus quality_history.jsonl (gefiltert)</div>
</div>
<div class="charts">
  <div class="chart wide">
    <div class="head"><h3>Input / Output / Cache-Tokens je Version</h3><div class="ax">Tokens</div></div>
    <p class="sub">
      <b>Cache-Tokens</b> werden g&uuml;nstiger abgerechnet.
      Filterung &auml;ndert die Datenbasis.
    </p>
    {"<div class='body' style='height:300px'><canvas id='chToken'></canvas></div><div class='legend' id='legToken'></div>" if not quality_empty else no_data}
  </div>
</div>

<!-- ===== DETAIL-TABELLE ===== -->
<div class="section-head">
  <h2>Detail-Tabelle</h2>
  <div class="rule"></div>
  <div class="note">sortierbar &middot; nach Filter</div>
</div>
<div class="table-wrap">
  <table class="detail-tbl" id="detailTable">
    <thead>
      <tr>
        <th data-col="note">Note <span class="sort-icon"></span></th>
        <th data-col="version">Version <span class="sort-icon"></span></th>
        <th data-col="pdf_short">PDF <span class="sort-icon"></span></th>
        <th class="num" data-col="hall">Hal-Rate % <span class="sort-icon"></span></th>
        <th class="num" data-col="cov">Coverage % <span class="sort-icon"></span></th>
        <th class="num" data-col="anchors_confirmed">Anchors OK <span class="sort-icon"></span></th>
      </tr>
    </thead>
    <tbody id="detailTableBody"></tbody>
  </table>
</div>

<!-- ===== ORIGINAL CHARTS ===== -->
<div class="section-head">
  <h2>Weitere Ansichten</h2>
  <div class="rule"></div>
  <div class="note">Log-basiert, nicht gefiltert</div>
</div>
<div class="charts">

  <!-- Chart 1: Akzeptanzrate -->
  <div class="chart">
    <div class="head"><h3>Akzeptanzrate je PDF</h3><div class="ax">in %</div></div>
    <p class="sub">
      Anteil der generierten Notes, die <b>automatisch akzeptiert</b> wurden &mdash;
      ohne manuelle Prüfung. Hohe Werte bedeuten, die Pipeline urteilt zielgenau.
    </p>
    {"<div class='body'><canvas id='ch1'></canvas></div>" if not accept_empty else no_data}
  </div>

  <!-- Chart 2: Scatter -->
  <div class="chart">
    <div class="head"><h3>Jede Note: Fehler gegen Abdeckung</h3><div class="ax">{len(scatter_chart.get("points",[]))} evaluierte Notes</div></div>
    <p class="sub">
      Jede Note ist ein Punkt. <b>Links oben</b> = ideal: wenige Fehler, hohe Abdeckung.
      Hover zeigt Titel und Quelle.
    </p>
    {f'{pdf_filter_html}<div class="body"><canvas id="ch2"></canvas></div><div class="legend" id="leg2"></div>' if not scatter_empty else no_data}
  </div>

  <!-- Chart 3: Skalierung -->
  <div class="chart">
    <div class="head"><h3>Skaliert die Pipeline mit der PDF-Laenge?</h3><div class="ax">Notes pro Run vs. Wortzahl</div></div>
    <p class="sub">
      Lange Papers erzeugen <b>nicht</b> linear mehr Notes &mdash; die Pipeline verdichtet,
      statt zu vervielfachen.
    </p>
    {"<div class='body'><canvas id='ch3'></canvas></div><div class='legend' id='leg3'></div>" if not scaling_empty else no_data}
  </div>

  <!-- Chart 4: Verlauf -->
  <div class="chart">
    <div class="head"><h3>Akzeptanzrate über Pipeline-Versionen</h3><div class="ax">Verlauf</div></div>
    <p class="sub">
      Wie sich die Qualität mit jeder Version verändert hat.
      <b>Steigende Linie</b> = Verbesserung.
    </p>
    {"<div class='body'><canvas id='ch4'></canvas></div><div class='legend' id='leg4'></div>" if not long_empty else no_data}
  </div>

  <!-- Chart 5: Tokens + Dauer (breit) -->
  <div class="chart wide">
    <div class="head"><h3>Laufzeit und Token-Verbrauch pro Run</h3><div class="ax">{kpis["total_runs"]} Runs &middot; chronologisch</div></div>
    <p class="sub">
      Wie lange ein Durchgang dauerte (<b>Linie</b>, rechte Achse in Minuten) und wie viele Tokens verbraucht wurden (<b>Balken</b>).
    </p>
    {"<div class='body' style='height:300px'><canvas id='ch5'></canvas></div><div class='legend' id='leg5'></div>" if not token_empty else no_data}
  </div>

</div>

<!-- Tabelle: PDF-Vergleich -->
<div class="section-head">
  <h2>Vergleich nach Quell-PDF</h2>
  <div class="rule"></div>
  <div class="note">&#9656; aufklappen für Quellendetails</div>
</div>
<div class="table-wrap">{pdf_table_html}</div>

<!-- Footer -->
<div class="foot">
  <div>Generiert <b style="color:var(--ink-2)">{generated_at}</b> &middot; Atomic Agent Eval &middot; lokal, nicht versioniert</div>
  <div><code>eval_dashboard.py</code></div>
</div>

</div><!-- .wrap -->

<script>
/* ---- Toggle für Tabellenzeilen ---- */
function toggleRow(btn) {{
  const detail = btn.closest("tr").nextElementSibling;
  if (!detail || !detail.classList.contains("detail-row")) return;
  const open = detail.style.display !== "none";
  detail.style.display = open ? "none" : "table-row";
  btn.classList.toggle("open", !open);
}}

/* ---- Chart.js Globals ---- */
const C = {{
  amber: '#e8b53b', teal: '#5bbfbf', coral: '#e07a5f',
  violet: '#8a86c8', mint: '#6dbf8c', slate: '#94a3b8',
  ink1: '#f1f5f9', ink2: '#cbd5e1', ink3: '#94a3b8',
  ink4: '#64748b', ink5: '#475569',
  grid: 'rgba(148,163,184,0.10)', bg: '#0f172a'
}};
Chart.defaults.color = C.ink4;
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.borderColor = C.grid;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.plugins.tooltip.backgroundColor = '#1c2742';
Chart.defaults.plugins.tooltip.titleColor = C.ink1;
Chart.defaults.plugins.tooltip.bodyColor = C.ink2;
Chart.defaults.plugins.tooltip.borderColor = 'rgba(148,163,184,0.2)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 4;
Chart.defaults.plugins.tooltip.titleFont = {{ weight: 600, size: 11.5 }};
Chart.defaults.plugins.tooltip.bodyFont = {{ size: 11.5 }};

const ACCEPT  = {accept_json};
const SCAT    = {scatter_json};
const LONG    = {long_json};
const TOKENS  = {token_json};
const SCALING = {scaling_json};
const QUALITY = {quality_json};

const axisCfg = (extra={{}}) => ({{
  grid: {{ color: C.grid, drawTicks: false }},
  border: {{ display: false }},
  ticks: {{ color: C.ink5, font: {{ size: 10.5 }}, padding: 8 }},
  ...extra
}});

/* ======================== FILTER ENGINE ======================== */
let _filterVersion = "";
let _filterPdf     = "";
let _sortCol       = "hall";
let _sortDir       = 1; // 1 = asc, -1 = desc

// Read URL params on load
(function() {{
  const p = new URLSearchParams(location.search);
  _filterVersion = p.get("version") || "";
  _filterPdf     = p.get("pdf") || "";
  const selVer = document.getElementById("filterVersion");
  const selPdf = document.getElementById("filterPdf");
  if (selVer && _filterVersion) selVer.value = _filterVersion;
  if (selPdf && _filterPdf)     selPdf.value = _filterPdf;
}})();

function getFilteredRows() {{
  return (QUALITY.rows || []).filter(r => {{
    if (_filterVersion && r.version !== _filterVersion) return false;
    if (_filterPdf     && r.pdf     !== _filterPdf)     return false;
    return true;
  }});
}}

function mean(arr) {{
  const v = arr.filter(x => x !== null && x !== undefined);
  return v.length ? v.reduce((a,b) => a+b, 0) / v.length : null;
}}

function colorClass(v, good, warn, invert) {{
  if (v === null || v === undefined) return "";
  if (invert) return v <= good ? "good" : (v >= warn ? "bad" : "warn");
  return v >= good ? "good" : (v <= warn ? "bad" : "warn");
}}

function updateKpiCards(rows) {{
  const halls = rows.map(r => r.hall).filter(x => x !== null);
  const covs  = rows.map(r => r.cov).filter(x => x !== null);
  const anch_conf  = rows.reduce((s,r) => s + (r.anchors_confirmed||0), 0);
  const anch_total = rows.reduce((s,r) => s + (r.anchors_total||0), 0);

  const avgHall = halls.length ? (halls.reduce((a,b)=>a+b,0)/halls.length).toFixed(1) : null;
  const avgCov  = covs.length  ? (covs.reduce((a,b)=>a+b,0)/covs.length).toFixed(1)  : null;
  const anchPct = anch_total   ? (100*anch_conf/anch_total).toFixed(1) : null;

  const hallEl = document.getElementById('kpiHall');
  const covEl  = document.getElementById('kpiCov');
  const nEl    = document.getElementById('kpiN');
  const anchEl = document.getElementById('kpiAnchors');

  if (hallEl) {{
    const cls = avgHall !== null ? colorClass(parseFloat(avgHall), 5, 15, true) : "";
    hallEl.className = 'v' + (cls ? ' '+cls : '');
    hallEl.innerHTML = avgHall !== null ? `${{avgHall}}<span class="unit">%</span>` : '--';
  }}
  if (covEl) {{
    const cls = avgCov !== null ? colorClass(parseFloat(avgCov), 50, 30, false) : "";
    covEl.className = 'v' + (cls ? ' '+cls : '');
    covEl.innerHTML = avgCov !== null ? `${{avgCov}}<span class="unit">%</span>` : '--';
  }}
  if (nEl)    nEl.textContent  = rows.length;
  if (anchEl) anchEl.innerHTML = anchPct !== null ? `${{anchPct}}<span class="unit">%</span>` : '--';

  const noteEl = document.getElementById('kpiNote');
  if (noteEl) {{
    const parts = [];
    if (_filterVersion) parts.push(_filterVersion);
    if (_filterPdf)     parts.push(_filterPdf.replace('.pdf','').split(' - ')[0]);
    noteEl.textContent = parts.length ? parts.join(' · ') : 'alle Notes';
  }}

  const countEl = document.getElementById('filterCount');
  if (countEl) countEl.textContent = `${{rows.length}} Notes`;
}}

/* ======================== SLOPE CHARTS ======================== */
let slopeChart1 = null;
let slopeChart2 = null;

function buildSlopeDatasets(metric) {{
  const versions = QUALITY.versions || [];
  return (QUALITY.slope_datasets || []).map(ds => {{
    const data = metric === 'hall' ? ds.hall_data : ds.cov_data;
    return {{
      label: ds.pdf_short,
      data: data,
      borderColor: ds.color,
      backgroundColor: ds.color,
      pointBackgroundColor: C.bg,
      pointBorderColor: ds.color,
      pointBorderWidth: 1.5,
      pointRadius: 4, pointHoverRadius: 6,
      borderWidth: 2, tension: 0.35, spanGaps: true,
    }};
  }});
}}

function buildSlopeDatasetsFiltered(metric) {{
  const rows    = getFilteredRows();
  const versions = QUALITY.versions || [];
  const pdfs = _filterPdf ? [_filterPdf] : (QUALITY.pdfs || []);
  const pdfColors = QUALITY.pdf_colors || {{}};

  return pdfs.map(pdf => {{
    const color = pdfColors[pdf] || C.slate;
    const pts = versions.map(v => {{
      const vrows = rows.filter(r => r.pdf === pdf && r.version === v);
      const vals  = vrows.map(r => metric === 'hall' ? r.hall : r.cov).filter(x => x !== null);
      if (!vals.length) return null;
      const sorted = [...vals].sort((a,b)=>a-b);
      return sorted[Math.floor(sorted.length/2)];
    }});
    const short = pdf.replace('.pdf','').split(' - ');
    const label = short.length >= 2 ? `${{short[0]}} (${{short[1]}})` : short[0];
    return {{
      label, data: pts,
      borderColor: color, backgroundColor: color,
      pointBackgroundColor: C.bg, pointBorderColor: color,
      pointBorderWidth: 1.5, pointRadius: 4, pointHoverRadius: 6,
      borderWidth: 2, tension: 0.35, spanGaps: true,
    }};
  }});
}}

function updateEndpointLabels(chart, metric) {{
  // Custom plugin: label last non-null point
  // handled inline via plugin below
}}

const endpointLabelPlugin = {{
  id: 'endpointLabels',
  afterDatasetsDraw(chart) {{
    const {{ctx, scales: {{x, y}}, data}} = chart;
    ctx.save();
    ctx.font = '500 10px Inter';
    ctx.textAlign = 'left';
    data.datasets.forEach((ds, di) => {{
      const meta = chart.getDatasetMeta(di);
      // Find last visible point
      let lastIdx = -1;
      for (let i = ds.data.length - 1; i >= 0; i--) {{
        if (ds.data[i] !== null && ds.data[i] !== undefined) {{ lastIdx = i; break; }}
      }}
      if (lastIdx < 0) return;
      const pt = meta.data[lastIdx];
      if (!pt) return;
      ctx.fillStyle = ds.borderColor;
      const val = ds.data[lastIdx];
      ctx.fillText(`${{val !== null ? val.toFixed(1) : ''}}%`, pt.x + 6, pt.y + 4);
    }});
    ctx.restore();
  }}
}};

if (!{str(quality_empty).lower()}) {{
  const versions = QUALITY.versions || [];

  // Slope Chart 1: Hal-Rate
  const sc1El = document.getElementById('chSlope1');
  if (sc1El) {{
    slopeChart1 = new Chart(sc1El, {{
      type: 'line',
      data: {{ labels: versions, datasets: buildSlopeDatasetsFiltered('hall') }},
      options: {{
        maintainAspectRatio: false, responsive: true,
        plugins: {{
          legend: {{ display: true, labels: {{ color: C.ink3, boxWidth:10, font:{{size:11}} }} }},
          tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.raw !== null ? ctx.raw.toFixed(1) : '--'}} %` }} }}
        }},
        scales: {{
          x: axisCfg({{ ticks: {{ color: C.ink4, maxRotation:45 }} }}),
          y: axisCfg({{ beginAtZero:true, max:100, ticks:{{ callback: v => v+'%', stepSize:25 }} }})
        }},
        animation: {{ duration:500 }}
      }},
      plugins: [endpointLabelPlugin]
    }});
    const leg = document.getElementById('legSlope1');
    if (leg) leg.innerHTML = (QUALITY.slope_datasets||[]).map(ds =>
      `<span><i style="background:${{ds.color}}"></i>${{ds.pdf_short}}</span>`
    ).join('');
  }}

  // Slope Chart 2: Coverage
  const sc2El = document.getElementById('chSlope2');
  if (sc2El) {{
    slopeChart2 = new Chart(sc2El, {{
      type: 'line',
      data: {{ labels: versions, datasets: buildSlopeDatasetsFiltered('cov') }},
      options: {{
        maintainAspectRatio: false, responsive: true,
        plugins: {{
          legend: {{ display: true, labels: {{ color: C.ink3, boxWidth:10, font:{{size:11}} }} }},
          tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.raw !== null ? ctx.raw.toFixed(1) : '--'}} %` }} }}
        }},
        scales: {{
          x: axisCfg({{ ticks: {{ color: C.ink4, maxRotation:45 }} }}),
          y: axisCfg({{ beginAtZero:true, max:100, ticks:{{ callback: v => v+'%', stepSize:25 }} }})
        }},
        animation: {{ duration:500 }}
      }},
      plugins: [endpointLabelPlugin]
    }});
    const leg = document.getElementById('legSlope2');
    if (leg) leg.innerHTML = (QUALITY.slope_datasets||[]).map(ds =>
      `<span><i style="background:${{ds.color}}"></i>${{ds.pdf_short}}</span>`
    ).join('');
  }}
}}

/* ======================== TOKEN CHART (gefiltert) ======================== */
let tokenQChart = null;
if (!{str(quality_empty).lower()}) {{
  const tokenEl = document.getElementById('chToken');
  if (tokenEl) {{
    const tokVer = QUALITY.versions || [];
    const tokData = QUALITY.token_by_ver || {{}};
    tokenQChart = new Chart(tokenEl, {{
      type: 'bar',
      data: {{
        labels: tokVer,
        datasets: [
          {{ label:'Input', data: tokVer.map(v => (tokData[v]||{{}}).tokens_input||0),
             backgroundColor: C.teal+'aa', stack:'t' }},
          {{ label:'Output', data: tokVer.map(v => (tokData[v]||{{}}).tokens_output||0),
             backgroundColor: C.mint+'aa', stack:'t' }},
          {{ label:'Cache', data: tokVer.map(v => (tokData[v]||{{}}).tokens_cache||0),
             backgroundColor: C.violet+'88', stack:'t' }},
        ]
      }},
      options: {{
        maintainAspectRatio: false, responsive: true,
        plugins: {{ legend: {{ display:true, labels:{{ color:C.ink3, boxWidth:10, font:{{size:11}} }} }} }},
        scales: {{
          x: axisCfg({{ grid:{{display:false}}, ticks:{{color:C.ink5}} }}),
          y: axisCfg({{ stacked:true, position:'left', title:{{display:true, text:'Tokens', color:C.ink4, font:{{size:11}}}} }})
        }},
        animation: {{ duration:400 }}
      }}
    }});
    const legT = document.getElementById('legToken');
    if (legT) legT.innerHTML = [
      {{col:C.teal,   lbl:'Input'}},
      {{col:C.mint,   lbl:'Output'}},
      {{col:C.violet, lbl:'Cache'}},
    ].map(x => `<span><i style="background:${{x.col}}"></i>${{x.lbl}}</span>`).join('');
  }}
}}

function updateTokenChart(rows) {{
  if (!tokenQChart) return;
  const versions = QUALITY.versions || [];
  const pdfColors = QUALITY.pdf_colors || {{}};

  const inp = [], out = [], cache = [];
  versions.forEach(v => {{
    const vrows = rows.filter(r => r.version === v);
    inp.push(vrows.reduce((s,r) => s+(r.tokens_input||0), 0));
    out.push(vrows.reduce((s,r) => s+(r.tokens_output||0), 0));
    cache.push(vrows.reduce((s,r) => s+(r.tokens_cache||0), 0));
  }});
  tokenQChart.data.datasets[0].data = inp;
  tokenQChart.data.datasets[1].data = out;
  tokenQChart.data.datasets[2].data = cache;
  tokenQChart.update();
}}

function updateSlopeCharts(rows) {{
  if (!slopeChart1 || !slopeChart2) return;
  const versions = QUALITY.versions || [];
  const pdfs = _filterPdf ? [_filterPdf] : (QUALITY.pdfs || []);
  const pdfColors = QUALITY.pdf_colors || {{}};

  function buildFiltered(metric) {{
    return pdfs.map(pdf => {{
      const color = pdfColors[pdf] || C.slate;
      const pts = versions.map(v => {{
        const vrows = rows.filter(r => r.pdf === pdf && r.version === v);
        const vals  = vrows.map(r => metric==='hall' ? r.hall : r.cov).filter(x => x!==null);
        if (!vals.length) return null;
        const sorted = [...vals].sort((a,b)=>a-b);
        return sorted[Math.floor(sorted.length/2)];
      }});
      const short = pdf.replace('.pdf','').split(' - ');
      const label = short.length >= 2 ? `${{short[0]}} (${{short[1]}})` : short[0];
      return {{
        label, data: pts,
        borderColor: color, backgroundColor: color,
        pointBackgroundColor: C.bg, pointBorderColor: color,
        pointBorderWidth: 1.5, pointRadius: 4, pointHoverRadius: 6,
        borderWidth: 2, tension: 0.35, spanGaps: true,
      }};
    }});
  }}

  slopeChart1.data.datasets = buildFiltered('hall');
  slopeChart2.data.datasets = buildFiltered('cov');
  slopeChart1.update();
  slopeChart2.update();
}}

/* ======================== DETAIL TABLE ======================== */
function hallColor(v) {{
  if (v === null || v === undefined) return '';
  return v <= 5 ? 'color:var(--good)' : v >= 15 ? 'color:var(--bad)' : 'color:var(--warn)';
}}
function covColor(v) {{
  if (v === null || v === undefined) return '';
  return v >= 50 ? 'color:var(--good)' : v <= 30 ? 'color:var(--bad)' : 'color:var(--warn)';
}}

function renderDetailTable(rows) {{
  const sorted = [...rows].sort((a,b) => {{
    let av = a[_sortCol], bv = b[_sortCol];
    if (av === null || av === undefined) av = _sortDir > 0 ? Infinity : -Infinity;
    if (bv === null || bv === undefined) bv = _sortDir > 0 ? Infinity : -Infinity;
    if (typeof av === 'string') return _sortDir * av.localeCompare(bv);
    return _sortDir * (av - bv);
  }});

  const body = document.getElementById('detailTableBody');
  if (!body) return;
  body.innerHTML = sorted.map(r => `
    <tr>
      <td class="note-name" title="${{r.note}}">${{r.note}}</td>
      <td><span class="tag cur">${{r.version}}</span></td>
      <td style="color:var(--ink-3);font-size:12px">${{r.pdf_short}}</td>
      <td class="num" style="${{hallColor(r.hall)}}">${{r.hall !== null ? r.hall.toFixed(1)+'%' : '&mdash;'}}</td>
      <td class="num" style="${{covColor(r.cov)}}">${{r.cov !== null ? r.cov.toFixed(1)+'%' : '&mdash;'}}</td>
      <td class="num" style="color:var(--ink-3)">${{r.anchors_confirmed}} / ${{r.anchors_total}}</td>
    </tr>
  `).join('');

  // Sort header indicators
  document.querySelectorAll('table.detail-tbl thead th').forEach(th => {{
    th.classList.remove('sort-asc','sort-desc');
    if (th.dataset.col === _sortCol) {{
      th.classList.add(_sortDir > 0 ? 'sort-asc' : 'sort-desc');
    }}
  }});
}}

// Table sort headers
document.querySelectorAll('table.detail-tbl thead th[data-col]').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (_sortCol === col) {{ _sortDir *= -1; }}
    else {{ _sortCol = col; _sortDir = 1; }}
    const rows = getFilteredRows();
    renderDetailTable(rows);
  }});
}});

/* ======================== FILTER LOGIC ======================== */
function applyFilters() {{
  const rows = getFilteredRows();
  updateKpiCards(rows);
  updateSlopeCharts(rows);
  updateTokenChart(rows);
  renderDetailTable(rows);

  // Persist in URL
  const params = new URLSearchParams();
  if (_filterVersion) params.set("version", _filterVersion);
  if (_filterPdf)     params.set("pdf",     _filterPdf);
  const qs = params.toString();
  history.replaceState(null, '', qs ? '?'+qs : location.pathname);
}}

document.getElementById('filterVersion')?.addEventListener('change', e => {{
  _filterVersion = e.target.value;
  applyFilters();
}});
document.getElementById('filterPdf')?.addEventListener('change', e => {{
  _filterPdf = e.target.value;
  applyFilters();
}});
document.getElementById('filterReset')?.addEventListener('click', () => {{
  _filterVersion = _filterPdf = "";
  const selVer = document.getElementById('filterVersion');
  const selPdf = document.getElementById('filterPdf');
  if (selVer) selVer.value = "";
  if (selPdf) selPdf.value = "";
  applyFilters();
}});

// Initial render
applyFilters();

/* ======================== ORIGINAL CHARTS ======================== */

/* ---- Chart 1: Akzeptanzrate Bar ---- */
if (!{str(accept_empty).lower()}) {{
  const colors = ACCEPT.colors;
  new Chart(document.getElementById('ch1'), {{
    type: 'bar',
    data: {{
      labels: ACCEPT.labels,
      datasets: [{{
        data: ACCEPT.values,
        backgroundColor: colors.map(c => c + 'cc'),
        borderColor: colors,
        borderWidth: 0,
        borderRadius: 2,
        barPercentage: 0.55,
        categoryPercentage: 0.85,
      }}]
    }},
    options: {{
      maintainAspectRatio: false, responsive: true,
      plugins: {{
        tooltip: {{ callbacks: {{ label: ctx => `${{ctx.parsed.y.toFixed(1)}} % akzeptiert` }} }}
      }},
      scales: {{
        x: axisCfg({{ grid: {{ display: false }}, ticks: {{ color: C.ink3, font: {{size:11}} }} }}),
        y: axisCfg({{ beginAtZero: true, max: 100, ticks: {{ callback: v => v + ' %', stepSize: 25 }} }})
      }},
      animation: {{ duration: 700, easing: 'easeOutCubic' }}
    }},
    plugins: [{{
      id: 'topLabels',
      afterDatasetsDraw(chart) {{
        const {{ctx, scales: {{x, y}}}} = chart;
        ctx.save();
        ctx.font = '600 12px Inter';
        ctx.textAlign = 'center';
        chart.getDatasetMeta(0).data.forEach((bar, i) => {{
          ctx.fillStyle = colors[i];
          ctx.fillText(ACCEPT.values[i].toFixed(1) + ' %', bar.x, bar.y - 8);
        }});
        ctx.restore();
      }}
    }}]
  }});
}}

/* ---- Chart 2: Scatter mit PDF-Filter ---- */
let scatterChart = null;
if (!{str(scatter_empty).lower()}) {{
  const COLORS_SC = [C.coral, C.teal, C.amber, C.violet, C.mint, C.slate];
  const pdfList = SCAT.pdfs || [];
  const pdfColor = {{}};
  pdfList.forEach((p, i) => pdfColor[p.raw] = COLORS_SC[i % COLORS_SC.length]);

  function buildDatasets(filter) {{
    const src = filter === "__all__" ? pdfList : pdfList.filter(p => p.raw === filter);
    return src.map((p) => ({{
      label: p.label,
      data: SCAT.points.filter(pt => pt.pdf === p.raw),
      backgroundColor: pdfColor[p.raw] + 'd0',
      borderColor: pdfColor[p.raw],
      borderWidth: 1,
      pointRadius: 7, pointHoverRadius: 9,
    }}));
  }}

  scatterChart = new Chart(document.getElementById('ch2'), {{
    type: 'scatter',
    data: {{ datasets: buildDatasets("__all__") }},
    options: {{
      maintainAspectRatio: false, responsive: true,
      plugins: {{
        legend: {{ display: true, labels: {{ color: C.ink3, boxWidth: 10, font: {{size:11}} }} }},
        tooltip: {{ callbacks: {{
          label: ctx => {{
            const p = ctx.raw;
            return [`${{p.label}}`, `Fehler: ${{p.x}} %  Abdeckung: ${{p.y}} %`];
          }}
        }} }}
      }},
      scales: {{
        x: axisCfg({{ min: 0, title: {{ display: true, text: 'Fehlerquote % (kleiner = besser)', color: C.ink4, font:{{size:11}} }}, ticks: {{ callback: v => v + ' %' }} }}),
        y: axisCfg({{ min: 0, title: {{ display: true, text: 'Abdeckung % (größer = besser)', color: C.ink4, font:{{size:11}} }}, ticks: {{ callback: v => v + ' %' }} }})
      }},
      animation: {{ duration: 700 }}
    }},
    plugins: [{{
      id: 'idealZone',
      beforeDatasetsDraw(chart) {{
        const {{ctx, scales: {{x, y}}}} = chart;
        const l = x.getPixelForValue(0), r = x.getPixelForValue({THRESH_HALL[0]});
        const t = y.getPixelForValue(y.max), b = y.getPixelForValue({THRESH_COV[1]});
        ctx.save();
        ctx.fillStyle = 'rgba(109,191,140,0.05)';
        ctx.fillRect(l, t, r-l, b-t);
        ctx.strokeStyle = 'rgba(109,191,140,0.25)';
        ctx.setLineDash([3,3]); ctx.lineWidth = 1;
        ctx.strokeRect(l, t, r-l, b-t);
        ctx.setLineDash([]);
        ctx.font = '600 10px Inter'; ctx.fillStyle = 'rgba(109,191,140,0.7)'; ctx.textAlign = 'left';
        ctx.fillText('IDEAL', l+6, t+14);
        ctx.restore();
      }}
    }}]
  }});

  const filterBar = document.getElementById('scatterFilter');
  if (filterBar) {{
    filterBar.addEventListener('click', e => {{
      const btn = e.target.closest('.filter-btn');
      if (!btn) return;
      filterBar.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      scatterChart.data.datasets = buildDatasets(btn.dataset.pdf);
      scatterChart.update();
    }});
  }}

  const leg2 = document.getElementById('leg2');
  if (leg2) {{
    leg2.innerHTML = pdfList.map(p =>
      `<span><i class="round" style="background:${{pdfColor[p.raw]}}"></i>${{p.label}}</span>`
    ).join('') +
    `<span style="margin-left:auto;color:var(--ink-5)"><i class="round" style="background:rgba(109,191,140,0.4)"></i>Zielzone</span>`;
  }}
}}

/* ---- Chart 3: Skalierung Scatter ---- */
if (!{str(scaling_empty).lower()}) {{
  const SCOL = {{ bates: C.coral, kuhlthau: C.teal, schlebbe: C.amber }};
  const allKeys = SCALING.keys || [];

  new Chart(document.getElementById('ch3'), {{
    type: 'scatter',
    data: {{
      datasets: allKeys.map(key => {{
        const pts = SCALING.points.filter(p => p.key === key);
        const col = SCOL[key] || C.slate;
        return {{
          label: (pts[0] || {{}}).label || key,
          data: pts.map(p => ({{ x: p.x, y: p.y, _p: p }})),
          backgroundColor: col + 'b0',
          borderColor: col, borderWidth: 1,
          pointRadius: 5, pointHoverRadius: 7,
        }};
      }})
    }},
    options: {{
      maintainAspectRatio: false, responsive: true,
      plugins: {{
        legend: {{ display: true, labels: {{ color: C.ink3, boxWidth: 10, font:{{size:11}} }} }},
        tooltip: {{ callbacks: {{ label: ctx => {{
          const p = ctx.raw._p || {{}};
          return [`${{p.label}} ${{p.ver}}`, `${{(p.x||0).toLocaleString('de-DE')}} Wörter · ${{p.y}} Notes (${{p.pct}} % akz.)`];
        }} }} }}
      }},
      scales: {{
        x: axisCfg({{ min: 0, title: {{ display: true, text: 'PDF-Laenge (Wörter)', color: C.ink4, font:{{size:11}} }}, ticks: {{ callback: v => (v/1000).toFixed(0)+'k' }} }}),
        y: axisCfg({{ min: 0, title: {{ display: true, text: 'Notes pro Run', color: C.ink4, font:{{size:11}} }}, ticks: {{ stepSize: 2 }} }})
      }},
      animation: {{ duration: 700 }}
    }}
  }});

  const leg3 = document.getElementById('leg3');
  if (leg3) leg3.innerHTML = allKeys.map(key => {{
    const col = SCOL[key] || C.slate;
    const lbl = (SCALING.points.find(p=>p.key===key)||{{}}).label || key;
    return `<span><i class="round" style="background:${{col}}"></i>${{lbl}}</span>`;
  }}).join('');
}}

/* ---- Chart 4: Longitudinal Line ---- */
if (!{str(long_empty).lower()}) {{
  new Chart(document.getElementById('ch4'), {{
    type: 'line',
    data: {{
      labels: LONG.versions,
      datasets: LONG.datasets.map(ds => {{
        const col = ds.color;
        return {{
          label: ds.label, data: ds.data,
          borderColor: col, backgroundColor: col,
          pointBackgroundColor: C.bg, pointBorderColor: col,
          pointBorderWidth: 1.5, pointRadius: 4, pointHoverRadius: 6,
          borderWidth: 2, tension: 0.35, spanGaps: true,
        }};
      }})
    }},
    options: {{
      maintainAspectRatio: false, responsive: true,
      plugins: {{
        legend: {{ display: true, labels: {{ color: C.ink3, boxWidth: 10, font:{{size:11}} }} }},
        tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.raw}} %` }} }}
      }},
      scales: {{
        x: axisCfg({{ ticks: {{ color: C.ink4, maxRotation: 45 }} }}),
        y: axisCfg({{ beginAtZero: true, max: 100, ticks: {{ callback: v => v+' %', stepSize: 25 }} }})
      }},
      animation: {{ duration: 700 }}
    }}
  }});

  const leg4 = document.getElementById('leg4');
  if (leg4) leg4.innerHTML = LONG.datasets.map(ds =>
    `<span><i style="background:${{ds.color}}"></i>${{ds.label}}</span>`
  ).join('');
}}

/* ---- Chart 5: Tokens + Dauer ---- */
if (!{str(token_empty).lower()}) {{
  new Chart(document.getElementById('ch5'), {{
    type: 'bar',
    data: {{
      labels: TOKENS.labels,
      datasets: [
        {{ type:'bar',  label:'Input-Tokens',  data:TOKENS.tokens_in,    backgroundColor:C.teal+'aa',   stack:'t', yAxisID:'yT' }},
        {{ type:'bar',  label:'Output-Tokens', data:TOKENS.tokens_out,   backgroundColor:C.mint+'aa',   stack:'t', yAxisID:'yT' }},
        {{ type:'bar',  label:'Cache-Tokens',  data:TOKENS.tokens_cache, backgroundColor:C.violet+'88', stack:'t', yAxisID:'yT' }},
        {{ type:'line', label:'Dauer (Min.)',   data:TOKENS.duration_min,
           borderColor:C.coral, backgroundColor:'transparent',
           borderWidth:2, pointRadius:3, yAxisID:'yD', tension:0.3,
           pointBackgroundColor:C.bg, pointBorderColor:C.coral, pointBorderWidth:1.5 }},
      ]
    }},
    options: {{
      maintainAspectRatio: false, responsive: true,
      plugins: {{ legend: {{ display:true, labels:{{ color:C.ink3, boxWidth:10, font:{{size:11}} }} }} }},
      scales: {{
        x: axisCfg({{ grid:{{display:false}}, ticks:{{color:C.ink5, maxRotation:45, font:{{size:9}}}} }}),
        yT: {{ ...axisCfg(), stacked:true, position:'left',  title:{{display:true, text:'Tokens', color:C.ink4, font:{{size:11}}}} }},
        yD: {{
          ...axisCfg(),
          position:'right', grid:{{drawOnChartArea:false}},
          ticks:{{ color:C.coral, callback: v => v+' min' }},
          title:{{display:true, text:'Dauer (Min.)', color:C.coral, font:{{size:11}}}},
        }}
      }},
      animation:{{ duration:700 }}
    }}
  }});

  const leg5 = document.getElementById('leg5');
  if (leg5) leg5.innerHTML = [
    {{col:C.teal,   lbl:'Input-Tokens'}},
    {{col:C.mint,   lbl:'Output-Tokens'}},
    {{col:C.violet, lbl:'Cache-Tokens'}},
    {{col:C.coral,  lbl:'Dauer (Min.)'}},
  ].map(x => `<span><i style="background:${{x.col}}"></i>${{x.lbl}}</span>`).join('');
}}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[dashboard] Lese Daten...")
    quality_rows = _read_quality_history()
    all_log_runs = _read_all_log_runs()
    token_runs   = _read_token_runs()
    log_data     = _build_log_data(all_log_runs)

    print(f"  quality_history: {len(quality_rows)} Eintraege")
    print(f"  Log-Runs: {len(all_log_runs)} aus {len(log_data)} PDFs")
    print(f"  Token-Runs: {len(token_runs)}")

    kpis           = _calc_kpis(log_data, all_log_runs, quality_rows, token_runs)
    pdf_table_rows = _calc_pdf_table(log_data, all_log_runs, quality_rows)
    accept_chart   = _chart_acceptance(log_data)
    scatter_chart  = _chart_scatter(quality_rows)
    long_chart     = _chart_longitudinal(log_data)
    token_chart    = _chart_tokens(token_runs)
    scaling_chart  = _chart_scaling(all_log_runs)
    quality_data   = _build_quality_chart_data(quality_rows)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = _build_html(
        kpis, pdf_table_rows,
        accept_chart, scatter_chart, long_chart, token_chart, scaling_chart,
        quality_data,
        generated_at,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\n[dashboard] Geschrieben: {OUTPUT}")

    webbrowser.open(OUTPUT.resolve().as_uri())
    print("[dashboard] Browser geöffnet.")


if __name__ == "__main__":
    main()
