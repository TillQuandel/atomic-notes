"""eval_dashboard.py -- Interaktives HTML-Dashboard Für Atomic-Agent Eval-Daten.

Liest:
  .cache/quality_history.jsonl    -- Stage-8-Eval pro Note
  .cache/eval/baseline/*.log      -- DRY-RUN Vault/Inbox-Routing-Logs
  .cache/runs/*.jsonl             -- Token- und Zeittracking pro LLM-Call

Schreibt: .cache/eval/dashboard.html  ->  oeffnet im Browser.

Usage: python eval_dashboard.py

STATUS (#98): Die Daten-/Aggregations-Funktionen hier (_calc_kpis, _build_log_data,
_chart_*, _PDF_META/_PDF_LABELS/THRESH_* etc.) sind weiterhin die maßgebliche
Quelle -- eval_dashboard_server.py importiert sie per `from generative import
eval_dashboard as D` fuer den Live-Server (Port 8051). NUR der Render-Pfad ganz
unten (_build_html + main(), Abschnitt "HTML zusammenbauen") ist Legacy: der
maßgebliche Render-Pfad ist eval_dashboard_server.py + internal/dashboard/
eval_dashboard.html. Der Legacy-Pfad hier hat 2026-06-19 eine Fehldiagnose
produziert (Dashboard-Filter-Refactor, "36 P1"-Regressionsverdacht kam vom Blick
auf dieses statische HTML statt auf den Live-Server). Direktaufruf `python
eval_dashboard.py` erzeugt weiterhin dieses (veraltete) statische Dashboard --
fuer den aktuellen Stand `eval_dashboard_server.py` starten.
"""

from __future__ import annotations

import json
import re
import statistics
import webbrowser
from datetime import datetime
from pathlib import Path

from generative.eval_common import wilson_ci

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / ".cache"
EVAL_DIR = CACHE_DIR / "eval" / "baseline"
RUNS_DIR = CACHE_DIR / "runs"
QUALITY_HISTORY = CACHE_DIR / "quality_history.jsonl"
OUTPUT = CACHE_DIR / "eval" / "dashboard.html"

_NOTE_RE = re.compile(r"^\s*\[DRY-RUN\] -> (Vault|Inbox)[^:]*: (.+?)\.md\b")
_MERGE_RE = re.compile(r"\[Merge-Stub\b|XSOURCE-MERGE")
_VER_RE = re.compile(r"_(v[\d.]+(?:\.\d+)*)(?:_run\d+)?\.log$")
_KEY_RE = re.compile(r"^([a-z]+)_")
_WORDS_RE = re.compile(r"(\d[\d.]*)\s+W")
_PAGES_RE = re.compile(r"(\d+)\s+S\.")
_CHUNKS_RE = re.compile(r"(\d+)\s+Chunks")
_NOTE_NS_PREFIX_RE = re.compile(r"^(?:vault|inbox|merge)__")

_PDF_LABELS: dict[str, str] = {
    "bates": "Bates 2017",
    "kuhlthau": "Kuhlthau ISP",
    "schlebbe": "Schlebbe & Greifeneder 2022",
}

# Schwellenwerte gut/ok/schlecht — belegte Richtwerte
# Quellen: Vectara Hallucination Leaderboard Mai 2026, RAGAS-Benchmarks
#   Fehlerquote:   Claude Sonnet-4.6 = 10.6 % (Vectara); gut <10 %, ok 10–20 %, schlecht >20 %
#   Abdeckung:     RAGAS Context Recall Ziel >0.80; akademische Texte: gut ≥50 %, ok 30–50 %
#   Akzeptanzrate: Knowledge Extraction 45–70 % normal; gut ≥70 %, ok 50–70 %
THRESH_ACCEPT = (85, 65)  # gut ≥85 %, ok 65–85 %, schlecht <65 %
THRESH_HALL = (5, 15)  # gut <5 %,  ok 5–15 %,  schlecht >15 %  — invert=True
THRESH_COV = (80, 50)  # gut ≥80 %, ok 50–80 %, schlecht <50 %

# Claude Design Farbpalette (editorial, kein Neon)
_PDF_COLORS: dict[str, str] = {
    "bates": "#e07a5f",  # coral
    "kuhlthau": "#5bbfbf",  # teal
    "schlebbe": "#e8b53b",  # amber
}
_COLOR_FALLBACKS = ["#8a86c8", "#6dbf8c", "#94a3b8"]

_PDF_META: dict[str, dict] = {
    "bates": {
        "titel": "Information Behavior",
        "autor": "Marcia J. Bates",
        "jahr": "2017",
        "in": "Encyclopedia of Library and Information Sciences, 3rd ed.",
        "thema": "Grundlagentext des Felds Information Behavior. Definiert Kernbegriffe "
        "(Information Seeking, Information Searching, Browsing), zeichnet die "
        "Begriffsgeschichte von Use Studies bis Information Behavior nach und "
        "stellt Bates' eigene Konzepte vor (Red Thread of Information, Berrypicking).",
        "sprache": "Englisch",
        "typ": "Handbuchkapitel / Überblicksartikel",
    },
    "kuhlthau": {
        "titel": "Information Search Process (ISP)",
        "autor": "Carol C. Kuhlthau",
        "jahr": "2009",
        "in": "Eigenständiges Dokument / Buchkapitel",
        "thema": "Beschreibt das ISP-Modell mit seinen 6 Phasen (Initiation, Selection, "
        "Exploration, Formulation, Collection, Presentation). Jede Phase umfasst "
        "drei Erfahrungsdimensionen: kognitiv, affektiv, physisch. "
        "Zentrale Konzepte: Uncertainty Principle, Zone of Intervention.",
        "sprache": "Englisch",
        "typ": "Theoriemodell-Dokument",
    },
    "schlebbe": {
        "titel": "Information Need, Informationsbedarf und -bedürfnis",
        "autor": "Kirsten Schlebbe & Elke Greifeneder",
        "jahr": "2022",
        "in": "Grundlagen der Informationswissenschaft (Kuhlen et al., Hrsg.)",
        "thema": "Deutschsprachiger Überblick über Konzepte des Informationsbedarfs. "
        "Behandelt Taylors Vier-Stufen-Typologie, Wilsons Modell (Information Needs "
        "als sekundäre Bedürfnisse), Greens Vier-Charakteristiken und "
        "Chatmans Small Worlds Theory.",
        "sprache": "Deutsch",
        "typ": "Handbuchkapitel",
    },
}

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _ver_sort_key(v: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", v))


def is_foss_version(version) -> bool:
    """True fuer nicht-generative Pipeline-Versionen. Realer Prefix ist `extractive-`
    (extractive/orchestrator.py: EXTRACTIVE_VERSION = "extractive-v0.2.0"); `foss-`
    wird als Forward-Compat-Alias mit erkannt. Diese Pipelines sind eine andere
    Architektur als die generative — ihre Versionen gehoeren nicht in denselben
    Versions-Trend (#36). NB: `foss-` allein matchte nichts Reales → Trennung war
    ein No-op (Cross-Model-Review 2026-06-23)."""
    return str(version or "").startswith(("extractive-", "foss-"))


def _latest_version(ver_map: dict) -> str:
    return sorted(ver_map.keys(), key=_ver_sort_key)[-1]


def _current_version(quality_rows: list[dict], current: str | None = None) -> str | None:
    """„Aktuelle" Pipeline-Version für die KPI-Kacheln (#191).

    Anker ist die installierte Code-Version (config.AGENT_VERSION). Höchste
    Nummer und jüngster Timestamp taugen beide nicht als Kriterium: verwaiste
    WIP-Branch-Zeilen im geteilten `.cache` tragen höhere Nummern als master
    (v0.3.141/142-Vorfall), Re-Evals alter Notes schreiben alte Versionen mit
    frischem Timestamp. Fallback, wenn zur Code-Version (noch) keine Zeilen
    existieren (frisch gebumpt): zeitlich jüngste generative Version mit Daten.
    """
    if current is None:
        try:
            from generative.config import AGENT_VERSION as current
        except Exception:
            current = None

    def _ver(r: dict) -> str:
        return r.get("version") or r.get("pipeline_version") or ""

    gen_rows = [r for r in quality_rows if _ver(r) and not is_foss_version(_ver(r))]
    if not gen_rows:
        return None
    if current and any(_ver(r) == current for r in gen_rows):
        return current
    return _ver(max(gen_rows, key=lambda r: r.get("timestamp") or ""))


def _capped_latest_version(versions: list[str], current: str | None) -> str | None:
    """Jüngste Log-Version, die nicht NEUER als die installierte Code-Version ist (#193).

    Fallback für `accept_ver`, wenn keine Eval-Zeilen die KPI-Version bestimmen.
    Reines Nummern-Max wählte verwaiste WIP-Branch-Versionen (#191-Muster):
    Versionen oberhalb der Code-Version sind per Definition nicht der Stand
    dieses Checkouts. `versions` muss per `_ver_sort_key` sortiert sein.
    Ohne bestimmbare Code-Version: unverändert Nummern-Max."""
    if not versions:
        return None
    if current:
        cur_key = _ver_sort_key(current)
        capped = [v for v in versions if _ver_sort_key(v) <= cur_key]
        return capped[-1] if capped else None
    return versions[-1]


def _median(lst: list[float]) -> float:
    # statistics.median (interpoliert) — muss zur kpi_trend-Berechnung im
    # Server passen, sonst zeigen KPI-Karte und Sparkline verschiedene Werte.
    return statistics.median(lst)


def _pooled_hall_stats(rows: list[dict]) -> dict | None:
    """Gepoolte Halluzinationsrate + Wilson-CI + Stichproben-Kennzahlen.

    Ankergewichtet statt notengewichtet — beantwortet „wie viel Prozent ALLER
    Anker sind falsch". Robuster als Median (kollabiert bei zero-inflation auf 0)
    und als Mean-of-rates (überbewertet kleine Notes). Eine Definition, geteilt
    von KPI-Kachel (_calc_kpis) und Sparkline (eval_dashboard_server), damit
    beide denselben Wert zeigen.

    Rückgabe (oder None, wenn keine bewertbaren Rows):
      pct           gepoolte Rate in Prozent
      ci_low/ci_high  95%-Wilson-CI in Prozent — NUR wenn Roh-Counts vorliegen,
                    sonst None (Mean-Fallback kann kein CI ausweisen)
      anchors_total Σ Anker (None im Mean-Fallback)
      rows_n        Zahl der bewertbaren Eval-Zeilen
      notes_n       distinct Notes darunter

    Fallback auf den Mittelwert der Pro-Note-Raten, wenn keine Roh-Counts
    vorliegen (historische DB-Rows vor 2026-06-27 hatten anchors_total nicht).
    """
    rate_rows = [r for r in rows if r.get("hallucination_rate") is not None and r["hallucination_rate"] >= 0]
    if not rate_rows:
        return None
    rows_n = len(rate_rows)
    notes_n = _distinct_notes(rate_rows)
    # Nur poolen, wenn ALLE bewertbaren Rows Roh-Counts haben. Sonst würde
    # ankergewichtet über eine Teilmenge gemittelt und der Rest still verworfen
    # (Cross-Review Codex+QWEN 2026-06-27) — bei gemischten Rows daher Mean.
    if all(r.get("anchors_total") is not None and r.get("anchors_hallucinated") is not None for r in rate_rows):
        th = sum(r["anchors_total"] for r in rate_rows)
        ah = sum(r["anchors_hallucinated"] for r in rate_rows)
        if th > 0:
            lo, hi = wilson_ci(ah, th)
            return {
                "pct": round(ah / th * 100, 1),
                "ci_low": round(lo * 100, 1),
                "ci_high": round(hi * 100, 1),
                "anchors_total": th,
                "rows_n": rows_n,
                "notes_n": notes_n,
            }
    # Mean-Fallback: ohne Roh-Counts kein CI ausweisen (null).
    return {
        "pct": round(statistics.mean(r["hallucination_rate"] for r in rate_rows) * 100, 1),
        "ci_low": None,
        "ci_high": None,
        "anchors_total": None,
        "rows_n": rows_n,
        "notes_n": notes_n,
    }


def _pooled_hall_pct(rows: list[dict]) -> float | None:
    """Gepoolte Halluzinationsrate in Prozent (nur der Wert; s. _pooled_hall_stats)."""
    stats = _pooled_hall_stats(rows)
    return stats["pct"] if stats else None


def _pdf_short_name(raw: str) -> str:
    name = raw.replace(".pdf", "").strip()
    parts = [p.strip() for p in name.split(" - ")]
    if len(parts) >= 2:
        return f"{parts[0]} ({parts[1]})" if parts[1].isdigit() else parts[0]
    name = re.sub(r"^\d+\.", "", name).strip()
    return name[:45]


def _top_versions(counts: dict, limit: int = 15, min_n: int = 3) -> list[str]:
    """Die `limit` neuesten Pipeline-Versionen mit mindestens `min_n` Eval-Notes,
    neueste zuerst. Hält Einzel-Note-Wegwerfläufe aus dem Versions-Filter (sonst
    füllen n=1-Versionen die Liste, während robuste ältere rausfallen).

    Die allerneueste Version ist **immer** dabei, auch wenn sie noch unter
    `min_n` liegt — ein frischer Lauf soll sofort sichtbar sein.
    """
    ranked = sorted(counts, key=_ver_sort_key, reverse=True)
    keep = [v for v in ranked if counts[v] >= min_n]
    if ranked and ranked[0] not in keep:
        keep.insert(0, ranked[0])
    return keep[:limit]


def orphan_versions(versions, current: str | None) -> list[str]:
    """Generative Pipeline-Versionen numerisch ÜBER `config.AGENT_VERSION` (#196 P1).

    Das ist die #191-Fehlerklasse (verwaiste WIP-Branch-Läufe kaperten die
    Anzeige) — sie meldet sich damit im Dashboard selbst. Hinweis-Charakter:
    bewusste Re-Evals/Branch-Läufe sind legitim. foss-/extractive-Versionen
    haben eine eigene Nummernwelt und bleiben außen vor.
    """
    if not current:
        return []
    cur_key = _ver_sort_key(current)
    out = {v for v in versions if v and not is_foss_version(v) and _ver_sort_key(v) > cur_key}
    return sorted(out, key=_ver_sort_key, reverse=True)


def _pdf_filter_key(raw: str) -> str:
    """„Bates - 2017" aus „Bates - 2017 - Information Behavior.pdf".

    Autor + Jahr (erste zwei „ - "-Teile). Bleibt ein startswith-Präfix des
    DB-`pdf`-Felds, sodass der bestehende Filter unverändert matcht, kürzt aber
    die verwirrend langen Titel und dedupliziert Varianten derselben Quelle.
    Verschiedene Jahre (Beutelspacher 2014 vs. 2022) bleiben getrennt.
    """
    name = raw.replace(".pdf", "").strip()
    parts = [p.strip() for p in name.split(" - ")]
    if len(parts) >= 2:
        return f"{parts[0]} - {parts[1]}"
    return parts[0]


def _pdf_slug(raw: str | None) -> str:
    """Kanonischer Vergleichsschlüssel für PDF-Bezeichner über alle Namensräume.

    Dieselbe Quelle liegt je nach Pipeline-Version als Volltitel
    („Bates - 2017 - Information Behavior.pdf", note_evals.pdf), Kurzlabel
    („Bates", pipeline_runs.pdf_label), Kebab-Key („bates-2017") oder
    Triple-Dash-Log-Key („bates---2017---information-behavior") vor (#202).
    Slug: lowercase, jeder Nicht-Alphanumerik-Lauf → ein „-".
    """
    if not raw:
        return ""
    name = str(raw).lower().removesuffix(".pdf")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def _pdf_matches(filter_value: str | None, *candidates: str | None) -> bool:
    """True, wenn ein Kandidat dieselbe Quelle wie der Filter-Wert bezeichnet.

    Präfix-Vergleich auf Slug-Ebene, nur an Segment-Grenzen („bates" matcht
    „bates-2017-…", nicht „batesworth-2020"). Richtungsoffen, weil mal der
    Filter (Volltitel aus note_evals), mal der Kandidat (Kurzlabel aus
    pipeline_runs) der längere Bezeichner ist (#202).
    """
    f = _pdf_slug(filter_value)
    if not f:
        return True
    for cand in candidates:
        c = _pdf_slug(cand)
        if c and (c == f or c.startswith(f + "-") or f.startswith(c + "-")):
            return True
    return False


# Sentinel für Quellen ohne bestimmbaren Gruppen-Schlüssel (leerer pdf-String,
# z. B. „.pdf"). Ein leerer Key darf NIE als `_pdf_matches`-Filter dienen —
# `_pdf_matches("")` matcht ALLES und zieht sonst jeden Routing-Run an sich (#194 P6).
_UNNAMED = "(unbenannt)"


def _pdf_group_key(raw: str | None) -> str:
    """Kanonischer Gruppen-Schlüssel für eine PDF-Quelle (SSoT mit dem PDF-Filter/
    -Dropdown, #202/#194).

    Autor-Jahr-Slug: `_pdf_slug(_pdf_filter_key(...))`. Kollabiert Volltitel
    („Bates - 2017 - Information Behavior.pdf") und Kebab-Variante („bates-2017")
    derselben Quelle auf denselben Schlüssel „bates-2017", hält aber verschiedene
    Jahre getrennt (Beutelspacher 2014 vs. 2022) — anders als der reine
    Segment-Präfix-Vergleich (`_pdf_matches`), der eine Autor-Variante
    transitiv über zwei Jahrgänge brücken würde. Kebab-Keys ohne „ - "-Struktur
    (Log-Namensraum, z. B. „test-short") bleiben vollständig erhalten.
    """
    return _pdf_slug(_pdf_filter_key(str(raw or "").replace(".pdf", "").strip()))


def _dedupe_pdf_options(labels) -> list[str]:
    """PDF-Dropdown-Optionen: Volltitel („Autor - Jahr - Titel") behalten, aber
    pro Quelle nur einmal — den vollständigsten Eintrag. Gruppiert wird auf
    Slug-Ebene, damit Kebab-Varianten („bates-2017") mit dem Volltitel
    zusammenfallen; eine reine Autor-Variante wird verworfen, wenn sie
    Segment-Präfix einer Autor-Jahr-Quelle ist (das verirrte „Bates" neben
    „Bates - 2017 - Information Behavior"). Der Filter matcht alle Varianten
    der Quelle über `_pdf_matches`.
    """
    by_key: dict[str, str] = {}
    for raw in labels:
        clean = raw.replace(".pdf", "").strip()
        if not clean:
            continue
        k = _pdf_group_key(clean)
        if k not in by_key or len(clean) > len(by_key[k]):
            by_key[k] = clean
    keys = sorted(by_key)
    drop = {k for k in keys if any(o != k and o.startswith(k + "-") for o in keys)}
    return [by_key[k] for k in keys if k not in drop]


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
        ver = _log_version(log) or "unknown"
        ext = _parse_log_extended(log)
        notes = ext["notes"]
        if not notes:
            continue
        n_total = len(notes)
        n_vault = sum(1 for v in notes.values() if v == "Vault")
        n_merge = sum(1 for v in notes.values() if v == "Merge")
        n_inbox = sum(1 for v in notes.values() if v == "Inbox")
        runs.append(
            {
                "key": key,
                "label": _PDF_LABELS.get(key, key),
                "ver": ver,
                "n_total": n_total,
                "n_vault": n_vault,
                "n_merge": n_merge,
                "n_inbox": n_inbox,
                # Creation-Rate: neue Notes → Vault
                "accept_pct": round(100 * n_vault / n_total, 1) if n_total else 0.0,
                # Enrichment-Rate: Merge-Stubs (korrekte Ergänzung bestehender Notes)
                "enrich_pct": round(100 * n_merge / n_total, 1) if n_total else 0.0,
                # Erfolgsrate: Vault + Merge-Stubs zusammen
                "success_pct": round(100 * (n_vault + n_merge) / n_total, 1) if n_total else 0.0,
                "words": ext["words"],
                "pages": ext["pages"],
                "chunks": ext["chunks"],
            }
        )
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
                # #197 Nachbesserung: Bookkeeping-/Event-Records (note_outcome,
                # anchor_stats, score_result, stage_outcome …) tragen kein `model`
                # (Schema-Invariante, vgl. _is_llm_call_record in eval_dashboard_server).
                # Ohne diesen Filter zählen sie als „Calls" → calls wird aufgebläht und
                # ein Run mit 0 echten LLM-Calls erschiene als 0-Token-Phantomzeile.
                if "model" not in r:
                    continue
                tin += r.get("input_tokens", 0) or 0
                tout += r.get("output_tokens", 0) or 0
                tcr += r.get("cache_read_tokens", 0) or 0
                tcw += r.get("cache_creation_tokens", 0) or 0
                dur_ms += r.get("duration_ms", 0) or 0
                count += 1
            except json.JSONDecodeError:
                pass
        if count > 0:
            # YYYYMMDD-HHMMSS → DD.MM HH:MM
            stem = jl.stem
            try:
                date_label = f"{stem[6:8]}.{stem[4:6]} {stem[9:11]}:{stem[11:13]}"
            except (IndexError, ValueError):
                date_label = stem
            runs.append(
                {
                    "date": date_label,
                    "run_id": stem,
                    "pdf_label": "",
                    "tokens_in": tin,
                    "tokens_out": tout,
                    "tokens_cache": tcr + tcw,
                    # #238: cache_read/cache_creation additiv getrennt — bisher nur
                    # als Summe (tokens_cache, oben, unveraendert fuer bestehende
                    # Konsumenten wie ch5) verfuegbar. Fuer die Billable-Aufschlues-
                    # selung im KPI-Kachel-Hint (in/out/cache_r/cache_c statt einer
                    # Summe, Issue-Fix-Vorschlag).
                    "tokens_cache_read": tcr,
                    "tokens_cache_create": tcw,
                    "duration_min": round(dur_ms / 60000, 1),
                    "calls": count,
                }
            )
    return runs


# ---------------------------------------------------------------------------
# Chart-Daten aufbereiten
# ---------------------------------------------------------------------------


def _calc_kpis(
    log_data: dict[str, dict[str, list[float]]],
    all_log_runs: list[dict],
    quality_rows: list[dict],
    token_runs: list[dict],
    current_version: str | None = None,
) -> dict:
    # KPIs = aktuelle Pipeline-Version (config-verankert, #191), nicht höchste
    # Nummer und nicht Durchschnitt aller Versionen. Einmal auflösen — auch
    # der accept_ver-Fallback unten braucht denselben Anker (#193).
    if current_version is None:
        try:
            from generative.config import AGENT_VERSION as current_version
        except Exception:
            current_version = None
    latest_pver = _current_version(quality_rows, current=current_version)
    latest_qrows = (
        [r for r in quality_rows if (r.get("version") or r.get("pipeline_version")) == latest_pver]
        if latest_pver
        else quality_rows
    )
    # Re-Eval-Dedup (Statistik-Review 2026-07-15): pro Note nur die neueste
    # Eval-Zeile — sonst poolen hall/cov unten Anker mehrfach-evaluierter
    # Notes mehrfach (Produktionsbeleg: 52 Zeilen / 40 distinct Notes bei
    # v0.3.140). Ab hier ist `latest_qrows` die EINE Basis für hall, cov UND
    # n_notes — die Kachel-n passt dadurch automatisch zur Pooling-Basis
    # (vorher: Kachel n=40, Pooling-Basis 52 — inkonsistent).
    latest_qrows = _dedup_latest_per_note(latest_qrows)

    all_versions = sorted({r["ver"] for r in all_log_runs if r.get("ver")}, key=_ver_sort_key)

    # Akzeptanz auf derselben Stichproben-Basis wie Fehlerquote/Belegrate
    # (neueste Pipeline-Version, gepoolt) — vorher mischte der Mittelwert
    # über die jeweils letzte Version JEDES PDFs alte und neue Versionen
    # unter der Überschrift "Qualität — <neueste Version>".
    # Kein nacktes Nummern-Max als Fallback: bei leeren quality_rows (z. B.
    # Filterkombination ohne Eval-Zeilen) kaperte sonst eine verwaiste
    # WIP-Version aus den Log-Runs die Akzeptanz-Kachel (#193, Audit-Fund).
    accept_ver = latest_pver or _capped_latest_version(all_versions, current_version)
    accept_runs = [r for r in all_log_runs if r.get("ver") == accept_ver]
    accept_generated = sum(r["n_total"] for r in accept_runs)
    # Vault-Notes der KPI-Version (nicht alle Vault-Notes) — Basis fuer den
    # Eval-Coverage-Hinweis: der Eval bewertet nur Vault-Notes.
    accept_vault = sum(r["n_vault"] for r in accept_runs)
    avg_accept = round(accept_vault / accept_generated * 100, 1) if accept_generated else None

    # Gepoolte Rate (ankergewichtet) statt Median: hallucination_rate ist
    # zero-inflated (>50 % der Notes haben 0 halluzinierte Anker), der Median
    # kollabierte sonst auf 0,0 % und verdeckte, dass das System halluziniert
    # (Bug 2026-06-27). Mean-Fallback für Rows ohne Roh-Counts.
    # #196 P4: dieselbe Poolung liefert zusätzlich das 95%-Wilson-CI + die
    # Anker-/Zeilen-Kennzahlen für die Kachel — im Mean-Fallback kein CI (null).
    hall_stats = _pooled_hall_stats(latest_qrows)
    avg_hall = hall_stats["pct"] if hall_stats else None

    cov_vals = [
        v for r in latest_qrows if (v := r.get("coverage_factual") or r.get("coverage_rate")) is not None and v >= 0
    ]
    avg_cov = round(_median(cov_vals) * 100, 1) if cov_vals else None
    total_generated = sum(r["n_total"] for r in all_log_runs)
    total_accepted = sum(r["n_vault"] for r in all_log_runs)
    total_merged = sum(r.get("n_merge", 0) for r in all_log_runs)
    # #198 P3: Lifetime-Summe nur über DB-gejointe Läufe (Option A). Waisen-
    # Traces (Trace-JSONL ohne pipeline_runs-Zeile) tragen kein `ver` und
    # blähten die KPI gegenüber dem Versions-Chart auf (`_chart_tokens_by_
    # version`, das über `ver` filtert). `db_matched` wird beim Server-Join
    # gesetzt; Default True erhält den deprecated Standalone-Pfad (main()/
    # _build_html), der _calc_kpis ohne Server-Join aufruft — dort existiert
    # der Key nie, und ohne DB-Join lässt sich „gejoint" nicht feststellen.
    total_tokens = sum(r["tokens_in"] + r["tokens_out"] for r in token_runs if r.get("db_matched", True))
    total_dur_s = sum(r["duration_min"] * 60 for r in token_runs if r.get("db_matched", True))
    latest_truns = [r for r in token_runs if r.get("ver") == latest_pver] if latest_pver else token_runs
    # cur_tokens (#238-Befund Wert 1): in+out ALLER Calls der aktuellen Version
    # inkl. Eval-Judges (token_runs kommt aus _read_token_runs(), das den
    # gesamten Trace liest, nicht nur die Pipeline-Phase) — OHNE Cache. Name/
    # Wert bewusst unveraendert (bestehende Konsumenten), nur das HTML-Label
    # wird praezisiert.
    cur_tokens = sum(r["tokens_in"] + r["tokens_out"] for r in latest_truns)
    # cur_tokens_cache: dieselbe token_runs-Quelle, Cache-Anteil separat (fuer
    # den Kachel-Hint) — tokens_cache war in token_runs bereits vorhanden, nur
    # nie in eine KPI-Summe eingerechnet.
    cur_tokens_cache = sum(r.get("tokens_cache", 0) or 0 for r in latest_truns)
    # Aufschluesselung statt nur der Summe (Issue-Fix-Vorschlag): in/out/
    # cache_read/cache_create separat fuer den Billable-Kachel-Hint.
    cur_tokens_in = sum(r.get("tokens_in", 0) or 0 for r in latest_truns)
    cur_tokens_out = sum(r.get("tokens_out", 0) or 0 for r in latest_truns)
    cur_tokens_cache_read = sum(r.get("tokens_cache_read", 0) or 0 for r in latest_truns)
    cur_tokens_cache_create = sum(r.get("tokens_cache_create", 0) or 0 for r in latest_truns)
    # cur_tokens_billable (#238-Fix): Billable als Leitzahl — in+out+cache,
    # exakt dieselbe Datenquelle wie cur_tokens (kein neuer Erfassungspfad,
    # nur die Cache-Spalte zusaetzlich aufsummiert). Entspricht #238-Befund
    # Wert 3 (echter billable Total inkl. Cache inkl. Eval), nur auf die
    # aktuelle Version statt den ganzen Trace gescoped (konsistent mit
    # cur_tokens/cur_dur_h daneben).
    cur_tokens_billable = cur_tokens + cur_tokens_cache
    # cur_dur_h = Summe der Call-Dauern (#239: "Agent-Rechenzeit", NICHT
    # Wall-Clock — bei parallelen Calls ueberlappt die Zeit, die Summe kann die
    # tatsaechlich vergangene Zeit deshalb ueber- ODER unterschaetzen, je nach
    # Parallelitaetsgrad; siehe #239-Befund: 0,98x-1,67x der echten Wall-Clock).
    cur_dur_h = round(sum(r["duration_min"] for r in latest_truns) / 60, 1)
    # cur_wall_h = echte Wall-Clock (#239-Fix): aus pipeline_runs.wall_clock_s,
    # das orchestrator.main() nach Abschluss von Stage-8 persistiert (vor dem
    # Fix nur die Call-Summe oder das VOR Stage-8 geschriebene duration_s,
    # beides keine Wall-Clock). Fehlt der Key (Alt-Traces vor #239 oder der
    # deprecated Standalone-Pfad ohne Server-Join), zaehlt der Run mit 0 statt
    # zu crashen — konsistent mit dem cost_usd-Default-Muster oben.
    cur_wall_h = round(sum(r.get("wall_clock_s", 0.0) or 0.0 for r in latest_truns) / 3600, 2)
    cur_cost_usd = round(sum(r.get("cost_usd", 0.0) or 0.0 for r in latest_truns), 4)

    # PDF-Zahl: distinct kanonische Quellen (SSoT mit der per-PDF-Tabelle, #194).
    # `len(log_data)` zählte pdf_key-Varianten mehrfach (bates/bates-2017/
    # bates---2017 = 3) UND verpasste Quellen, die nur in quality_rows stehen
    # (Hertzum/Kaletski). Union aus Eval- und Routing-only-Quellen.
    # Leere Gruppen-Keys ausschließen: sie zählten als Geister-Quelle UND
    # `_pdf_matches("")` matchte jeden Routing-Run → _log_gk kollabierte (#194 P6).
    _eval_gk = {gk for r in quality_rows if r.get("pdf") and (gk := _pdf_group_key(r["pdf"]))}
    _log_gk = {
        gk
        for lr in all_log_runs
        if not any(_pdf_matches(egk, lr.get("label"), lr.get("key")) for egk in _eval_gk)
        and (gk := _pdf_group_key(lr.get("label") or lr.get("key")))
    }
    n_canonical_pdfs = len(_eval_gk | _log_gk)

    # "Evaluierte Notes"-Quote (#224): Zähler (n_notes, oben) und Nenner müssen
    # dieselbe Versions-Basis teilen, sonst entstehen Werte >100 % (Zähler aus
    # DB/note_evals früh gefiltert, alter Nenner `total_generated` aus
    # all_log_runs über ALLE Versionen gepoolt). Nenner hier = all_log_runs
    # exakt der kpi_version (`latest_pver`, derselbe Anker wie `latest_qrows`
    # oben) — bei explizitem Versions-Filter sind quality_rows/all_log_runs
    # bereits serverseitig auf dieselbe Version eingeschränkt, ungefiltert
    # erzwingt der exakte Vergleich unten dieselbe Beschränkung nachträglich.
    generated_kpi_ver = latest_pver
    kpi_gen_runs = [r for r in all_log_runs if r.get("ver") == generated_kpi_ver]
    kpi_generated = sum(r["n_total"] for r in kpi_gen_runs)
    _n_notes = _distinct_notes(latest_qrows)
    # Guard statt Absurd-Wert: Nenner 0/fehlend (keine Log-/DB-Zeile zur
    # Version) oder Zähler > Nenner (Datenlücke, z. B. Re-Evals ohne
    # zugehörigen Log-Run) → null, Client zeigt "–" statt >100 %.
    notes_eval_pct = round(100 * _n_notes / kpi_generated, 1) if kpi_generated and _n_notes <= kpi_generated else None

    return {
        "avg_accept": avg_accept,
        "avg_hall": avg_hall,
        # #196 P4: 95%-Wilson-CI der gepoolten Fehlerquote (in Prozent) plus
        # Anker-/Zeilen-/Notes-N für die Kachel-Sub-Info. CI ist None, wenn die
        # Rate über den Mean-Fallback (Rows ohne Roh-Counts) berechnet wurde.
        "hall_ci_low": hall_stats["ci_low"] if hall_stats else None,
        "hall_ci_high": hall_stats["ci_high"] if hall_stats else None,
        "hall_anchors_total": hall_stats["anchors_total"] if hall_stats else None,
        "hall_rows_n": hall_stats["rows_n"] if hall_stats else None,
        "hall_notes_n": hall_stats["notes_n"] if hall_stats else None,
        "avg_cov": avg_cov,
        "kpi_accept_n": accept_generated,
        "kpi_vault_n": accept_vault,
        # Version-Basis von Akzeptanz/Vault-Zahl — kann von kpi_version
        # abweichen, wenn (noch) keine Eval-Rows zur neuesten Version existieren
        "kpi_accept_ver": accept_ver,
        "kpi_version": latest_pver,
        # distinct Notes, nicht Eval-Instanzen (#194 #4): Re-Evals derselben Note
        # blähten die Zahl auf (50 Instanzen / 39 Notes) und gewichteten die
        # Poolung implizit nach Testlauf-Häufigkeit. Fehlt ein Note-Identifier
        # (synthetische Rows), zählt der Laufindex jede Zeile einzeln.
        "n_notes": _n_notes,
        # Server-berechnete "% der Notes"-Quote + ihre Versions-Basis (#224) —
        # Client rechnet nicht mehr selbst aus n_notes/total_generated (die
        # unterschiedliche Versions-/Filterkontexte hatten). null = "–".
        "notes_eval_pct": notes_eval_pct,
        "generated_kpi_ver": generated_kpi_ver,
        "total_runs": len(all_log_runs),
        "n_pdfs": n_canonical_pdfs,
        "n_versions": len(all_versions),
        "versions_range": f"{all_versions[0]}–{all_versions[-1]}"
        if len(all_versions) > 1
        else (all_versions[0] if all_versions else "--"),
        "total_generated": total_generated,
        "total_accepted": total_accepted,
        "total_merged": total_merged,
        "total_dropped": sum(r.get("n_dropped", 0) or 0 for r in all_log_runs),
        "total_tokens": total_tokens,
        "total_dur_h": round(total_dur_s / 3600, 1),
        "cur_tokens": cur_tokens,
        "cur_tokens_cache": cur_tokens_cache,
        "cur_tokens_billable": cur_tokens_billable,
        "cur_tokens_in": cur_tokens_in,
        "cur_tokens_out": cur_tokens_out,
        "cur_tokens_cache_read": cur_tokens_cache_read,
        "cur_tokens_cache_create": cur_tokens_cache_create,
        "cur_dur_h": cur_dur_h,
        "cur_wall_h": cur_wall_h,
        "cur_cost_usd": cur_cost_usd,
    }


def _row_version(r: dict) -> str:
    return r.get("version") or r.get("pipeline_version") or ""


def _distinct_notes(rows: list[dict]) -> int:
    """Distinct-Note-Zahl einer Row-Menge: `note_path`|`note` als Identität,
    Fallback Zeilenindex (synthetische Rows ohne Identifier zählen einzeln).

    SSoT für die „Evaluierte Notes"-KPI-Kachel (`_calc_kpis`), ihre per-Version-
    Sparkline (`kpi_trend["n"]`, Server) und die per-PDF-Tabelle — sonst zählt
    die Kachel distinct (40) und die Sparkline Eval-Instanzen (52) (#194 #4)."""
    return len({r.get("note_path") or r.get("note") or i for i, r in enumerate(rows)})


def _note_key(r: dict, idx: int) -> str:
    """Normalisierter Note-Schlüssel für Dedup/Identitätsvergleiche.

    `note_path`/`note` mit gestripptem Namespace-Prefix (`vault__`/`inbox__`/
    `merge__`) — eine Note kann zwischen zwei Re-Evals den Namespace wechseln
    (Routing-Änderung/Merge), ein reiner Feldvergleich würde dieselbe Note
    sonst als zwei Identitäten zählen (Statistik-Review 2026-07-15). Fehlt ein
    Identifier (synthetische Rows), ist der Schlüssel der Zeilenindex —
    konsistent mit dem Fallback in `_distinct_notes`."""
    raw = r.get("note_path") or r.get("note")
    if not raw:
        return f"__row{idx}"
    return _NOTE_NS_PREFIX_RE.sub("", str(raw))


def _dedup_latest_per_note(rows: list[dict]) -> list[dict]:
    """Pro (normalisierter) Note NUR die neueste Eval-Zeile.

    Statistik-Review 2026-07-15 (3 unabhängige Opus-Statistiker, konvergent +
    adversarial bestätigt): note_evals enthält mehrere Zeilen derselben Note
    innerhalb einer pipeline_version (Re-Evals + identische Duplikat-Inserts;
    Produktionsbeleg v0.3.140 = 52 Zeilen / 40 distinct Notes, 12 Duplikate).
    Ungefiltert poolt jede KPI-Aggregation (`_pooled_hall_stats`, Coverage-
    Median, `kpi_trend` im Server) Anker mehrfach-evaluierter Notes mehrfach —
    Pseudoreplikation, ~2pp Bias auf der gepoolten Fehlerquote (oft
    re-evaluierte Notes haben tendenziell gute Raten, die die Poolung sonst
    nach unten ziehen).

    Aufrufer MÜSSEN `rows` vorher auf eine einzelne pipeline_version
    einschränken — der Dedup hier ist versionsblind (reine Note-Identität);
    das Vermischen mehrerer Versionen ist Aufgabe des Callers, nicht dieser
    Funktion.

    „Neueste" = größter `timestamp`-String (ISO 8601, lexikographisch
    sortierbar); Tie-Break `eval_id` (ebenfalls Timestamp-präfixiert), dann
    Listenposition — rein für Determinismus bei exaktem Timestamp+eval_id-
    Gleichstand (Rows ohne beides: die letzte in der Liste gewinnt, was zur
    `ORDER BY timestamp`-Reihenfolge von `db.query_note_evals` passt). Rows
    ohne Note-Identifier zählen einzeln (s. `_note_key`)."""
    best: dict[str, tuple] = {}
    order: list[str] = []
    for i, r in enumerate(rows):
        key = _note_key(r, i)
        sort_key = (str(r.get("timestamp") or ""), str(r.get("eval_id") or ""), i)
        prev = best.get(key)
        if prev is None or sort_key > prev[0]:
            if prev is None:
                order.append(key)
            best[key] = (sort_key, r)
    return [best[k][1] for k in order]


def _newest_capped_version(versions: list[str], current: str | None) -> str | None:
    """Neueste Version, die nicht NEUER als die Config-Version ist (#191);
    fällt auf die neueste vorhandene zurück, wenn ALLE Versionen Orphans sind
    (statt None wie `_capped_latest_version` — die Zeile soll sichtbar bleiben).
    `versions` muss per `_ver_sort_key` sortiert sein."""
    return _capped_latest_version(versions, current) or (versions[-1] if versions else None)


def _accept_from_runs(runs: list[dict], prefer_ver: str | None, current: str | None) -> tuple:
    """Akzeptanz (gepoolt Σvault/Σtotal) aus den Routing-Runs einer Quelle.

    `accept` stammt aus pipeline_runs/log — einer ANDEREN Quelle als hall/cov
    (note_evals trägt keine acceptance_status). Bevorzugt die Eval-Version der
    Zeile (`prefer_ver`), damit alle Kennzahlen auf derselben Version stehen;
    fehlt sie in den Runs, die neueste (gekappte) Routing-Version. Gibt
    (accept, accept_ver, accept_n, n_merge) zurück; accept_ver kann von der
    Eval-Version abweichen und wird darum getrennt ausgewiesen.

    n_merge (#249): Σ Merge-Stub-Notes derselben Runs/Version — SSoT-konform
    auf derselben Versions-Basis wie accept/accept_n berechnet (kein zweiter
    Aggregations-Pfad). Speist die "gemergt, nicht evaluiert"-Kategorie im
    Scatter-Panel; diese Notes landen im Vault, werden aber nie separat neu
    evaluiert (#237-Diagnosebefund).
    """
    if not runs:
        return None, None, 0, 0
    vers = sorted({r["ver"] for r in runs if r.get("ver")}, key=_ver_sort_key)
    av = prefer_ver if (prefer_ver and prefer_ver in vers) else _newest_capped_version(vers, current)
    at = [r for r in runs if r.get("ver") == av]
    tot = sum(r.get("n_total", 0) or 0 for r in at)
    vault = sum(r.get("n_vault", 0) or 0 for r in at)
    n_merge = sum(r.get("n_merge", 0) or 0 for r in at)
    accept = round(vault / tot * 100, 1) if tot else None
    return accept, av, len(at), n_merge


def _calc_pdf_table(
    log_data: dict[str, dict[str, list[float]]],
    all_log_runs: list[dict],
    quality_rows: list[dict],
    current_version: str | None = None,
) -> list[dict]:
    """Eine Zeile je kanonischer Quell-PDF, EINE Datengrundlage (#194).

    Zuvor mischte jede Zeile drei Quellen: version/accept aus den Routing-Logs,
    hall/cov über ALLE Versionen gepoolt (Substring-Match), n_notes =
    Eval-Instanzen. Ergebnis: Bates-Dreifachzeilen (pdf_key-Drift), fehlende
    PDFs (nur in quality_rows) und version-gemischte Kennzahlen.

    Jetzt: kanonische Gruppierung (`_pdf_group_key`, SSoT mit dem PDF-Filter),
    Iteration über die Union aus Eval- und Routing-Quelle, hall/cov/n_notes aus
    GENAU der neuesten Eval-Version der Quelle (gekappt gegen die Config-Version,
    #191), n_notes = distinct Notes. accept kommt weiterhin aus dem Routing
    (getrennt via accept_ver/accept_n ausgewiesen).
    """
    if current_version is None:
        try:
            from generative.config import AGENT_VERSION as current_version
        except Exception:
            current_version = None

    # ── Eval-Gruppen (Primärquelle) ──
    # Leere Gruppen-Keys unter dem Sentinel isolieren, nie als Match-Filter nutzen
    # (`_pdf_matches("")` matcht ALLES, #194 P6).
    eval_groups: dict[str, list[dict]] = {}
    for r in quality_rows:
        if r.get("pdf"):
            eval_groups.setdefault(_pdf_group_key(r["pdf"]) or _UNNAMED, []).append(r)
    group_keys = [gk for gk in eval_groups if gk != _UNNAMED]

    # ── Routing-Runs eindeutig EINER Gruppe zuordnen; Rest = Routing-only ──
    # Ein jahrloser Autor-Key („beutelspacher") ist Segment-Präfix MEHRERER
    # Jahrgangs-Gruppen (2014 UND 2022). Ihn per max(key=len)-Tie einer Gruppe
    # zuzuschlagen war order-abhängig und willkürlich (#194 P2). Regel (ehrlich,
    # ohne Raten): exakter kanonischer Key gewinnt; sonst nur bei GENAU einem
    # Match zuordnen; bei mehrdeutigem Kurz-Key eigene Routing-only-Zeile.
    log_by_group: dict[str, list[dict]] = {}
    log_only: dict[str, list[dict]] = {}
    for lr in all_log_runs:
        run_key = _pdf_group_key(lr.get("label") or lr.get("key")) or _UNNAMED
        if run_key != _UNNAMED and run_key in eval_groups:
            log_by_group.setdefault(run_key, []).append(lr)  # exakter Gruppen-Key
            continue
        matched = [gk for gk in group_keys if _pdf_matches(gk, lr.get("label"), lr.get("key"))]
        if len(matched) == 1:
            log_by_group.setdefault(matched[0], []).append(lr)  # eindeutiger Präfix-Match
        else:
            # 0 Matches → echte Routing-only-Quelle; ≥2 Matches → mehrdeutiger
            # Kurz-Key: nicht einer Jahrgangs-Gruppe zuschlagen, eigene Zeile.
            log_only.setdefault(run_key, []).append(lr)

    def _label_for(group_key: str, qrows: list[dict], fallback: str) -> str:
        if group_key in _PDF_LABELS:
            return _PDF_LABELS[group_key]
        pdfs = [r.get("pdf") for r in qrows if r.get("pdf")]
        return _pdf_short_name(max(pdfs, key=len)) if pdfs else _PDF_LABELS.get(fallback, fallback)

    def _words_for(runs: list[dict]) -> int | None:
        vals = [r["words"] for r in runs if r.get("words")]
        return int(_median(vals)) if vals else None

    rows = []
    # 1) Gruppen mit Eval-Daten
    for gk in sorted(eval_groups):
        qrows = eval_groups[gk]
        vers = sorted({_row_version(r) for r in qrows if _row_version(r)}, key=_ver_sort_key)
        ver = _newest_capped_version(vers, current_version)
        # Orphan: ALLE Versionen der Quelle sind NEUER als die Config-Version
        # (#191). _newest_capped_version fällt dann auf versions[-1] (Orphan)
        # zurück — die Zeile soll das sichtbar kennzeichnen, nicht unmarkiert als
        # „neueste Eval-Version" zeigen (#194 P5).
        orphan = bool(vers) and _capped_latest_version(vers, current_version) is None
        at = [r for r in qrows if _row_version(r) == ver]
        # Re-Eval-Dedup (Statistik-Review 2026-07-15): dieselbe Basis wie die
        # KPI-Kachel (`_calc_kpis`) — sonst zeigen Kachel und per-PDF-Tabelle
        # unterschiedliche gepoolte Raten für dieselbe Version.
        at = _dedup_latest_per_note(at)
        n_notes = _distinct_notes(at)
        hall = _pooled_hall_pct(at)
        cov_vals = [v for r in at if (v := r.get("coverage_factual") or r.get("coverage_rate")) is not None and v >= 0]
        cov = round(_median(cov_vals) * 100, 1) if cov_vals else None
        runs = log_by_group.get(gk, [])
        accept, accept_ver, accept_n, n_merge = _accept_from_runs(runs, ver, current_version)
        rows.append(
            {
                "key": gk,
                "label": _UNNAMED if gk == _UNNAMED else _label_for(gk, qrows, gk),
                "version": ver,
                "orphan": orphan,
                "routing_only": False,
                "accept": accept,
                "accept_ver": accept_ver,
                "accept_n": accept_n,
                # n_merge (#249): Merge-Stubs derselben Runs/Version — nie separat
                # neu evaluiert, speist die "gemergt, nicht evaluiert"-Kategorie.
                "n_merge": n_merge,
                "hall": hall,
                "cov": cov,
                "n_notes": n_notes,
                "words": _words_for(runs),
            }
        )
    # 2) Routing-only-Quellen (Union): Akzeptanz vorhanden, keine Eval-Daten.
    # Das „Version (Eval)"-Feld trägt hier die Routing-Version — der Client muss
    # das über routing_only unterscheiden (#194 P7).
    for gk in sorted(log_only):
        runs = log_only[gk]
        accept, accept_ver, accept_n, n_merge = _accept_from_runs(runs, None, current_version)
        label = _PDF_LABELS.get(gk) or next((r.get("label") for r in runs if r.get("label")), gk)
        rvers = sorted({r["ver"] for r in runs if r.get("ver")}, key=_ver_sort_key)
        orphan = bool(rvers) and _capped_latest_version(rvers, current_version) is None
        rows.append(
            {
                "key": gk,
                "label": label,
                "version": accept_ver,
                "orphan": orphan,
                "routing_only": True,
                "accept": accept,
                "accept_ver": accept_ver,
                "accept_n": accept_n,
                "n_merge": n_merge,
                "hall": None,
                "cov": None,
                "n_notes": 0,
                "words": _words_for(runs),
            }
        )
    return rows


def _chart_acceptance(pdf_rows: list[dict]) -> dict:
    """Akzeptanz-Balken aus den kanonischen per-PDF-Zeilen (SSoT mit der Tabelle,
    #194). Trägt `n` (distinct Eval-Notes je Balken), damit der ins-quality-
    Streifen „0 % von n Notes" von „0 Notes" unterscheiden kann."""
    labels, values, colors, n = [], [], [], []
    for r in pdf_rows:
        if r.get("accept") is None:
            continue
        labels.append(r.get("label") or r.get("key"))
        values.append(r["accept"])
        colors.append(_pdf_color(r.get("key", "")))
        n.append(r.get("n_notes", 0) or 0)
    return {"labels": labels, "values": values, "colors": colors, "n": n}


def _chart_scatter(quality_rows: list[dict]) -> dict:
    points: list[dict] = []
    pdf_map: dict[str, str] = {}
    for r in quality_rows:
        hall = r.get("hallucination_rate")
        cov = r.get("coverage_factual") or r.get("coverage_rate")
        if hall is None or cov is None or float(hall) < 0 or float(cov) < 0:
            continue
        label = r.get("note") or r.get("note_title") or "?"
        label = re.sub(r"^(vault|inbox)__", "", label).replace(".md", "")
        pdf = r.get("pdf") or r.get("source_pdf") or "unbekannt"
        if pdf not in pdf_map:
            pdf_map[pdf] = _pdf_short_name(pdf)
        points.append(
            {
                "x": round(float(hall) * 100, 1),
                "y": round(float(cov) * 100, 1),
                "label": label,
                "pdf": pdf,
                "pdf_label": pdf_map[pdf],
            }
        )
    pdfs = [{"raw": k, "label": v} for k, v in pdf_map.items()]
    return {"points": points, "pdfs": pdfs}


def _chart_longitudinal(log_data: dict) -> dict:
    all_ver: set[str] = set()
    for vm in log_data.values():
        all_ver.update(vm.keys())
    versions = sorted(all_ver, key=_ver_sort_key)
    datasets = []
    for key in sorted(log_data):
        vm = log_data[key]
        data_pts = [_median(vm[v]) if vm.get(v) else None for v in versions]
        # Anzahl Runs je Punkt (#204 P8c): ein Punkt aus n=1 Run ist duenn —
        # ein einzelner 0%-Run kann eine ganze Version wie einen Einbruch
        # aussehen lassen. Client zeigt das im Tooltip statt es zu verstecken.
        n_pts = [len(vm[v]) if vm.get(v) else 0 for v in versions]
        datasets.append(
            {
                "label": _PDF_LABELS.get(key, key),
                "data": data_pts,
                "n": n_pts,
                "color": _pdf_color(key),
            }
        )
    return {"versions": versions, "datasets": datasets}


_DELTA_MIN_N = 20  # unter N=20 kein Besser/Schlechter-Urteil (Apophenie-Schutz)

# Corpus-Overlap-Guard (Statistik-Review 2026-07-15, 3 unabhängige Opus-
# Statistiker, konvergent + adversarial bestätigt): n>=20 in beiden Versionen
# allein härtet ein Delta NICHT gegen PDF-Mix-Artefakte. Produktionsbeleg
# v0.3.140 -> v0.3.143: beide n>=20 (40/22 distinct Notes), aber nur 3 von 9/5
# PDFs geteilt — von den 22 Notes der neueren Version stammen nur 8 (36 %) aus
# einer PDF, die auch in v0.3.140 vorkommt. Das dort gemessene +2,7pp-Hall-
# Delta ist damit größtenteils ein ausgetauschter Corpus, kein echter
# Versions-Effekt, wurde vor diesem Fix aber als "reliable" (grün/rot) gezeigt.
# Schwelle 50 %: unter der Hälfte notengewichteter Quellen-Überlappung ist der
# Corpus faktisch ausgetauscht. Als Konstante konfigurierbar statt hart
# inline verdrahtet, falls sich das in der Praxis als zu streng/lax zeigt.
_DELTA_MIN_PDF_OVERLAP = 0.5


def version_delta(kpi_trend: dict, metric: str) -> dict:
    """Delta der neuesten Version gegen die letzte belastbare Vorversion.

    `kpi_trend["versions"]` ist aufsteigend sortiert (neueste = letzte Position),
    die Metrik-Arrays laufen parallel dazu.

    #196 P5: Vergleichsbasis ist die jüngste FRÜHERE Version mit einem
    vorhandenen Metrik-Wert UND n>=_DELTA_MIN_N — nicht starr die direkte
    Vorversion. Direkte Nachbarversionen sind oft Einzel-Note-Wegwerfläufe
    (n=1–2), gegen die jedes Delta reliable:false wäre. Existiert KEINE frühere
    Version mit n>=_DELTA_MIN_N, greift der bisherige Fallback (direkte
    Vorversion, Chip bleibt grau/reliable:false). `prev_version`/`prev_n` machen
    im Client-Tooltip transparent, WOGEGEN verglichen wird.

    Statistik-Review 2026-07-15: `reliable` ist zusätzlich an den Corpus-
    Overlap gekoppelt (`_DELTA_MIN_PDF_OVERLAP`) — der Notes-Anteil der
    neuesten Version, dessen PDF-Quelle auch in der Vergleichsversion
    vorkommt. `kpi_trend["pdf_notes"]` (optional, vom Server befüllt, s.
    eval_dashboard_server.py) trägt dafür je Version ein
    `{pdf_group_key: n_notes}`-Dict. Fehlt der Key (ältere Aufrufer/Tests ohne
    `pdf_notes`), greift der Guard nicht — reines n>=20-Verhalten bleibt
    rückwärtskompatibel. `reason` unterscheidet im Rückgabewert, WARUM
    `reliable` False ist (`"n_lt_20"` vs. `"pdf_mix"`), damit der Client die
    beiden Fälle unterschiedlich betexten kann ("n<20" vs. "nicht vergleichbar
    (PDF-Mix)").
    """
    values = kpi_trend.get(metric) or []
    ns = kpi_trend.get("n") or []
    versions = kpi_trend.get("versions") or []
    pdf_notes = kpi_trend.get("pdf_notes") or []
    latest = values[-1] if values else None
    n_latest = ns[-1] if ns else None

    def _n_at(i: int) -> int:
        return (ns[i] if i < len(ns) else 0) or 0

    def _ver_at(i: int):
        return versions[i] if i < len(versions) else None

    def _pdf_notes_at(i: int) -> dict:
        return pdf_notes[i] if i < len(pdf_notes) and pdf_notes[i] else {}

    # Jüngste frühere Version mit Wert und n>=_DELTA_MIN_N suchen (rückwärts).
    prev = prev_version = n_prev = prev_idx = None
    for i in range(len(values) - 2, -1, -1):
        if values[i] is not None and _n_at(i) >= _DELTA_MIN_N:
            prev, n_prev, prev_version, prev_idx = values[i], _n_at(i), _ver_at(i), i
            break
    else:
        # Fallback: direkte Vorversion (bisheriges Verhalten; reliable bleibt False).
        if len(values) >= 2:
            prev = values[-2]
            n_prev = ns[-2] if len(ns) >= 2 else None
            prev_version = _ver_at(len(values) - 2)
            prev_idx = len(values) - 2

    delta = None if (latest is None or prev is None) else round(latest - prev, 4)
    n_reliable = delta is not None and (n_latest or 0) >= _DELTA_MIN_N and (n_prev or 0) >= _DELTA_MIN_N

    # Notengewichteter Corpus-Overlap: Anteil der Notes der NEUESTEN Version,
    # deren PDF-Quelle auch in der Vergleichsversion (prev_idx) vorkommt.
    pdf_overlap = None
    if prev_idx is not None:
        latest_pdf_notes = _pdf_notes_at(len(values) - 1)
        prev_pdf_notes = _pdf_notes_at(prev_idx)
        total = sum(latest_pdf_notes.values())
        if total:
            shared = sum(n for pdf, n in latest_pdf_notes.items() if pdf in prev_pdf_notes)
            pdf_overlap = round(shared / total, 3)
    pdf_ok = pdf_overlap is None or pdf_overlap >= _DELTA_MIN_PDF_OVERLAP

    reliable = n_reliable and pdf_ok
    reason = None
    if delta is not None:
        if not n_reliable:
            reason = "n_lt_20"
        elif not pdf_ok:
            reason = "pdf_mix"

    return {
        "latest": latest,
        "prev": prev,
        "delta": delta,
        "reliable": reliable,
        "prev_version": prev_version,
        "prev_n": n_prev,
        "pdf_overlap": pdf_overlap,
        "reason": reason,
    }


def _chart_tokens(runs: list[dict]) -> dict:
    return {
        "labels": [r["date"] for r in runs],
        "pdf_labels": [r.get("pdf_label", "") for r in runs],
        "tokens_in": [r["tokens_in"] for r in runs],
        "tokens_out": [r["tokens_out"] for r in runs],
        "tokens_cache": [r["tokens_cache"] for r in runs],
        "duration_min": [r["duration_min"] for r in runs],
    }


_SCALING_RECENT_KEEP = 10  # juengste N Versionen ungedimmt (#36 P2)


def mark_scaling_recency(points: list[dict], keep: int = _SCALING_RECENT_KEEP) -> list[dict]:
    """Gibt eine neue Punktliste zurueck, in der die Punkte der juengsten `keep`
    Versionen `recent=True` tragen, aeltere `recent=False`. So kann der Client
    kaputte Frueh-Versions-Aeren dimmen, statt sie ungefiltert in die
    PDF-Laengen-Skalierung zu mischen (#36 P2). Mutiert die Eingabe nicht.
    """
    versions = sorted({p["ver"] for p in points if p.get("ver")}, key=_ver_sort_key)
    recent_set = set(versions[-keep:]) if keep > 0 else set()
    return [{**p, "recent": p.get("ver") in recent_set} for p in points]


def _chart_tokens_by_version(runs: list[dict]) -> dict:
    """Token-Komposition (Summe) + Median-Duration pro Pipeline-Version,
    aufsteigend sortiert (neueste rechts), foss-frei. Ersetzt die chronologische
    Pro-Run-Achse, die bei vielen Runs unlesbar war und keinen Vergleich trug
    (#36, E6)."""
    by_ver: dict = {}
    for r in runs:
        ver = r.get("ver") or r.get("pipeline_version")
        if not ver or is_foss_version(ver):
            continue
        b = by_ver.setdefault(ver, {"in": 0, "out": 0, "cache": 0, "dur": []})
        b["in"] += r.get("tokens_in", 0) or 0
        b["out"] += r.get("tokens_out", 0) or 0
        b["cache"] += r.get("tokens_cache", 0) or 0
        if r.get("duration_min") is not None:
            b["dur"].append(r["duration_min"])
    versions = sorted(by_ver, key=_ver_sort_key)
    return {
        "labels": versions,
        "tokens_in": [by_ver[v]["in"] for v in versions],
        "tokens_out": [by_ver[v]["out"] for v in versions],
        "tokens_cache": [by_ver[v]["cache"] for v in versions],
        "duration_min": [round(_median(by_ver[v]["dur"]), 1) if by_ver[v]["dur"] else None for v in versions],
    }


def _chart_scaling(all_log_runs: list[dict]) -> dict:
    points = [
        {
            "x": r["words"],
            "y": r["n_total"],
            "y_vault": r["n_vault"],
            "pages": r["pages"],
            "key": r["key"],
            "label": r["label"],
            "ver": r["ver"],
            "pct": r["accept_pct"],
        }
        for r in all_log_runs
        if r["words"] is not None
    ]
    points = mark_scaling_recency(points)
    keys = sorted({p["key"] for p in points})
    return {"points": points, "keys": keys}


def _build_quality_chart_data(quality_rows: list[dict]) -> dict:
    """Bereitet quality_history-Daten fuer Client-seitige Charts und Filter auf."""
    rows_clean = []
    for r in quality_rows:
        note = r.get("note") or r.get("note_title") or "?"
        note = re.sub(r"^(vault|inbox)__", "", note).replace(".md", "")
        pdf = r.get("pdf") or r.get("source_pdf") or "unbekannt"
        ver = r.get("version") or "unknown"
        hall = r.get("hallucination_rate")
        cov = r.get("coverage_factual") or r.get("coverage_rate")
        anch_total = r.get("anchors_total") or 0
        anch_conf = r.get("anchors_confirmed") or 0
        rows_clean.append(
            {
                "note": note,
                "pdf": pdf,
                "pdf_short": _pdf_short_name(pdf),
                "version": ver,
                "hall": round(float(hall) * 100, 1) if hall is not None else None,
                "cov": round(float(cov) * 100, 1) if cov is not None else None,
                "anchors_confirmed": anch_conf,
                "anchors_total": anch_total,
                "tokens_input": r.get("tokens_input", 0) or 0,
                "tokens_output": r.get("tokens_output", 0) or 0,
                "tokens_cache": r.get("tokens_cache_read", 0) or 0,
                "wall_time_s": r.get("wall_time_s", 0) or 0,
                "small_sample": r.get("small_sample_warning", False),
            }
        )

    all_versions = sorted({r["version"] for r in rows_clean}, key=_ver_sort_key)
    all_pdfs = sorted({r["pdf"] for r in rows_clean})

    # Slope-Daten: Median Hall-Rate + Coverage pro Version pro PDF
    slope_datasets = []
    pdf_colors_used: dict[str, str] = {}
    for i, pdf in enumerate(all_pdfs):
        key = (
            _KEY_RE.match(pdf.replace(".pdf", "").strip()).group(1)
            if _KEY_RE.match(pdf.replace(".pdf", "").strip())
            else pdf.lower()[:6]
        )
        color = _pdf_color(key, i)
        pdf_colors_used[pdf] = color
        hall_pts = []
        cov_pts = []
        for v in all_versions:
            vrows = [r for r in rows_clean if r["pdf"] == pdf and r["version"] == v and r["hall"] is not None]
            hall_pts.append(_median([r["hall"] for r in vrows]) if vrows else None)
            vrows2 = [r for r in rows_clean if r["pdf"] == pdf and r["version"] == v and r["cov"] is not None]
            cov_pts.append(_median([r["cov"] for r in vrows2]) if vrows2 else None)
        slope_datasets.append(
            {
                "pdf": pdf,
                "pdf_short": _pdf_short_name(pdf),
                "color": color,
                "hall_data": hall_pts,
                "cov_data": cov_pts,
            }
        )

    # Token-Daten: Summe Input/Output/Cache pro Version
    token_by_ver: dict[str, dict] = {}
    for v in all_versions:
        vrows = [r for r in rows_clean if r["version"] == v]
        token_by_ver[v] = {
            "tokens_input": sum(r["tokens_input"] for r in vrows),
            "tokens_output": sum(r["tokens_output"] for r in vrows),
            "tokens_cache": sum(r["tokens_cache"] for r in vrows),
            "n": len(vrows),
        }

    return {
        "rows": rows_clean,
        "versions": all_versions,
        "pdfs": all_pdfs,
        "pdf_colors": pdf_colors_used,
        "slope_datasets": slope_datasets,
        "token_by_ver": token_by_ver,
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
    return f'<span class="mini-bar {cls}"><i style="width:{min(value, 100):.0f}%"></i></span>'


def _render_pdf_table(rows: list[dict]) -> str:
    if not rows:
        return '<p style="color:var(--ink-4);font-style:italic;padding:2rem 0">Keine Daten vorhanden.</p>'

    header = (
        '<table class="cmp"><thead><tr>'
        '<th style="width:28px"></th>'
        "<th>Quell-PDF</th>"
        "<th>Letzte Version</th>"
        '<th class="num">W&ouml;rter</th>'
        '<th class="num">Akzeptiert</th>'
        '<th class="num">Fehlerquote</th>'
        '<th class="num">Abdeckung</th>'
        '<th class="num">Eval.&nbsp;Notes</th>'
        "</tr></thead><tbody>"
    )
    body = ""
    for r in rows:
        key = r.get("key", "")
        meta = _PDF_META.get(key, {})
        w_str = f"{r['words']:,}".replace(",", ".") if r["words"] else "&mdash;"
        acc_q = _quant(r["accept"], THRESH_ACCEPT[0], THRESH_ACCEPT[1])
        hall_q = _quant(r["hall"], THRESH_HALL[0], THRESH_HALL[1], invert=True)
        cov_q = _quant(r["cov"], THRESH_COV[0], THRESH_COV[1])
        bar = _mini_bar(r["accept"], THRESH_ACCEPT[0], THRESH_ACCEPT[1])
        ver = r.get("version") or "&mdash;"
        # Routing-only-/Orphan-Zeilen kennzeichnen (Legacy-CLI, #194 P5/P7)
        if r.get("routing_only"):
            ver = f"{ver} (nur Routing)"
        elif r.get("orphan"):
            ver = f"&#9888; {ver}"
        ver_cls = "cur" if (not r.get("routing_only") and not r.get("orphan") and r.get("version")) else ""

        if meta:

            def _dl(k: str, v: str) -> str:
                return f'<div class="pdf-dl-row"><dt>{k}</dt><dd>{v}</dd></div>'

            meta_html = (
                '<dl class="pdf-meta">'
                + _dl("Vollst. Titel", meta.get("titel", "--"))
                + _dl("Autor(en)", meta.get("autor", "--"))
                + _dl("Jahr", meta.get("jahr", "--"))
                + _dl("Erschienen in", meta.get("in", "--"))
                + _dl("Sprache / Typ", f"{meta.get('sprache', '--')} &middot; {meta.get('typ', '--')}")
                + f'<div class="pdf-dl-row pdf-dl-full"><dt>Inhalt</dt><dd>{meta.get("thema", "--")}</dd></div>'
                + "</dl>"
            )
            toggle = (
                '<td style="text-align:center;width:28px">'
                '<button class="expand-btn" onclick="toggleRow(this)" title="Details">&#9656;</button>'
                "</td>"
            )
            detail = (
                f'<tr class="detail-row" style="display:none">'
                f'<td colspan="8" style="padding:0;border-bottom:1px solid var(--hairline)">'
                f"{meta_html}</td></tr>"
            )
        else:
            toggle = "<td></td>"
            detail = ""

        body += (
            f'<tr class="data-row">'
            f"{toggle}"
            f'<td class="td-name" onclick="toggleRow(this.closest(\'tr\').querySelector(\'.expand-btn\'))" style="cursor:pointer">{r["label"]}</td>'
            f'<td><span class="tag {ver_cls}">{ver}</span></td>'
            f'<td class="num" style="color:var(--ink-3)">{w_str}</td>'
            f'<td class="num">{bar}{acc_q}</td>'
            f'<td class="num">{hall_q}</td>'
            f'<td class="num">{cov_q}</td>'
            f'<td class="num" style="color:var(--ink-3)">{r["n_notes"] or "&mdash;"}</td>'
            f"</tr>"
            f"{detail}"
        )
    return header + body + "</tbody></table>"


# ---------------------------------------------------------------------------
# HTML zusammenbauen -- LEGACY-RENDERPFAD (#98)
# ---------------------------------------------------------------------------
# Ab hier: statischer Einmal-Render fuer `python eval_dashboard.py`. Der
# maßgebliche Render-Pfad ist eval_dashboard_server.py (Live-Server, Port 8051)
# + internal/dashboard/eval_dashboard.html -- NICHT dieser hier. Dieser Pfad hat
# 2026-06-19 eine Fehldiagnose produziert (Regressionsverdacht "36 P1" kam vom
# Blick auf dieses veraltete statische HTML). Nicht loeschen ohne die
# main()-CLI-Nutzung (`python eval_dashboard.py`) vorher abzuloesen.

_CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
_FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Inter+Tight:wght@500;600;700;800&display=swap"


def _build_html(
    kpis: dict,
    pdf_table_rows: list[dict],
    accept_chart: dict,
    scatter_chart: dict,
    long_chart: dict,
    token_chart: dict,
    scaling_chart: dict,
    quality_data: dict,
    generated_at: str,
) -> str:
    """LEGACY (#98): baut das statische Einmal-Dashboard fuer `python eval_dashboard.py`.

    Maßgeblich ist eval_dashboard_server.py + internal/dashboard/eval_dashboard.html,
    NICHT dieser Renderpfad -- siehe Modul-Docstring. Hat 2026-06-19 eine
    Fehldiagnose produziert.
    """
    accept_json = json.dumps(accept_chart, ensure_ascii=False)
    scatter_json = json.dumps(scatter_chart, ensure_ascii=False)
    long_json = json.dumps(long_chart, ensure_ascii=False)
    token_json = json.dumps(token_chart, ensure_ascii=False)
    scaling_json = json.dumps(scaling_chart, ensure_ascii=False)
    quality_json = json.dumps(quality_data, ensure_ascii=False)

    accept_empty = not accept_chart.get("labels")
    scatter_empty = not scatter_chart.get("points")
    long_empty = not long_chart.get("versions")
    token_empty = not token_chart.get("labels")
    scaling_empty = not scaling_chart.get("points")
    quality_empty = not quality_data.get("rows")

    # Scatter-Filterleiste (Chart 2)
    pdf_filter_html = ""
    if not scatter_empty and scatter_chart.get("pdfs"):
        btns = '<button class="filter-btn active" data-pdf="__all__">Alle PDFs</button>'
        for p in scatter_chart["pdfs"]:
            btns += f'<button class="filter-btn" data-pdf="{p["raw"]}">{p["label"]}</button>'
        pdf_filter_html = f'<div class="filter-bar" id="scatterFilter">{btns}</div>'

    tok_m = f"{kpis['total_tokens'] / 1_000_000:.2f}M" if kpis["total_tokens"] else "--"

    pdf_table_html = _render_pdf_table(pdf_table_rows)

    no_data = (
        '<p style="color:var(--ink-4);font-style:italic;padding:3rem 0;text-align:center">Keine Daten vorhanden.</p>'
    )

    # Versions- und PDF-Optionen fuer Filter-Dropdowns
    all_versions_opts = "".join(f'<option value="{v}">{v}</option>' for v in quality_data.get("versions", []))
    all_pdfs_opts = "".join(f'<option value="{p}">{_pdf_short_name(p)}</option>' for p in quality_data.get("pdfs", []))

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
  <div class="cell"><div class="k">Akzeptiert</div><div class="v">{kpis["total_accepted"]}<small> / {round(100 * kpis["total_accepted"] / kpis["total_generated"], 1) if kpis["total_generated"] else "--"}&nbsp;%</small></div></div>
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
    <div class="head"><h3>Jede Note: Fehler gegen Abdeckung</h3><div class="ax">{len(scatter_chart.get("points", []))} evaluierte Notes</div></div>
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
    token_runs = _read_token_runs()
    log_data = _build_log_data(all_log_runs)

    print(f"  quality_history: {len(quality_rows)} Eintraege")
    print(f"  Log-Runs: {len(all_log_runs)} aus {len(log_data)} PDFs")
    print(f"  Token-Runs: {len(token_runs)}")

    kpis = _calc_kpis(log_data, all_log_runs, quality_rows, token_runs)
    pdf_table_rows = _calc_pdf_table(log_data, all_log_runs, quality_rows)
    accept_chart = _chart_acceptance(pdf_table_rows)
    scatter_chart = _chart_scatter(quality_rows)
    long_chart = _chart_longitudinal(log_data)
    token_chart = _chart_tokens(token_runs)
    scaling_chart = _chart_scaling(all_log_runs)
    quality_data = _build_quality_chart_data(quality_rows)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = _build_html(
        kpis,
        pdf_table_rows,
        accept_chart,
        scatter_chart,
        long_chart,
        token_chart,
        scaling_chart,
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
