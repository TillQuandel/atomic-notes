"""Anzahl-basierte Rotation der Prozess-Caches (#151, Punkt 6).

`.cache/llm` (LLM-Response-Cache; fresh-run-Namespaces werden nie wieder
getroffen) und `.cache/runs` (Trace-Datei pro Lauf) wachsen monoton — beide
werden am Run-Ende auf eine konservative Datei-Obergrenze gestutzt, aelteste
zuerst (Muster: `gui/run_history.prune_old_records`).

Bewusst NICHT angefasst: `quality_history.jsonl` (Eval-Daten, longitudinal
ausgewertet) und `.bak`-Dateien (werden nie geloescht).
"""

from __future__ import annotations

import logging
from pathlib import Path

from generative.config import CACHE_DIR, CACHE_LLM_MAX_FILES, CACHE_RUNS_MAX_FILES

logger = logging.getLogger(__name__)


def prune_cache_dir(cache_dir: Path | str, keep: int) -> int:
    """Loescht die aeltesten Dateien in `cache_dir`, wenn mehr als `keep` vorhanden
    sind. Nur regulaere Dateien direkt im Verzeichnis; `.bak` bleibt immer erhalten.
    Gibt die Anzahl geloeschter Dateien zurueck."""
    cache_dir = Path(cache_dir)
    if keep < 0 or not cache_dir.exists():
        return 0
    files = [p for p in cache_dir.iterdir() if p.is_file() and p.suffix != ".bak"]
    if len(files) <= keep:
        return 0
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # neueste zuerst
    deleted = 0
    for p in files[keep:]:
        try:
            p.unlink()
            deleted += 1
        except OSError as exc:
            logger.warning("Konnte alte Cache-Datei nicht loeschen (%s): %s", p, exc)
    return deleted


def rotate_run_caches(cache_dir: Path | str = CACHE_DIR) -> tuple[int, int]:
    """Rotiert `.cache/llm` und `.cache/runs` auf ihre Obergrenzen. Gibt
    (geloeschte_llm, geloeschte_runs) zurueck."""
    cache_dir = Path(cache_dir)
    n_llm = prune_cache_dir(cache_dir / "llm", CACHE_LLM_MAX_FILES)
    n_runs = prune_cache_dir(cache_dir / "runs", CACHE_RUNS_MAX_FILES)
    return n_llm, n_runs
