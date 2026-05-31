"""Stage-6-Crash-Report-Writer (Issue #17).

Bei einer Stage-6-Exception wird der unverifizierte Draft verworfen (nicht geschrieben)
und stattdessen ein strukturierter JSON-Report nach .cache/failed/<run-id>/<slug>.json
abgelegt — damit der Crash diagnostizierbar bleibt statt in einer stderr-Zeile zu verschwinden.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _slug(title: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", (title or "").lower()).strip("-")
    return s[:80] or "untitled"


def write_crash_report(failed_dir: Path, payload: dict) -> Path:
    """Schreibt payload als UTF-8-JSON nach failed_dir/<slug(title)>.json.

    Legt failed_dir bei Bedarf an. Gibt den Pfad der geschriebenen Datei zurück.
    """
    failed_dir.mkdir(parents=True, exist_ok=True)
    path = failed_dir / f"{_slug(payload.get('title', ''))}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
