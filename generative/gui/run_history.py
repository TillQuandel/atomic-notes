"""Run-Historie fuer die Live-GUI (P4): vergangene Laeufe bleiben einsehbar.

Reine Datei-/Record-Logik (kein FastAPI, kein Subprocess) — leicht test- und
wiederverwendbar. Ein Record ist ein flaches JSON-Objekt pro Lauf:

    {run_id, started_at, finished_at, source_pdf, dry_run, options, rc, notes}

`notes` ist bereits die fertig aggregierte P3-Item-Liste (siehe
`generative.gui.app._output_items_from_events`) — dieses Modul dupliziert die
Aggregation nicht, es nimmt sie fertig entgegen.

Records liegen unter `generative/.cache/gui/runs/<run_id>.json` (Cache-
Artefakt, kein Vault-Inhalt — L4-Whitelist gilt trotzdem: `run_id` wird strikt
gegen `RUN_ID_RE` geprueft, Pfade nur per `joinpath` unterhalb `runs_dir`).

`duration_s`/`tokens` (P5) sind optional — nur vorhanden, wenn der Lauf ein
`run_summary`-Event erzeugt hat (Final-Report Zeit/Tokens, s. run_parser.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Strikt genug, um Traversal (".." enthaelt einen Punkt) und Pfadtrenner
# (`/`, `\`) von vornherein auszuschliessen — kein zusaetzlicher Sonderfall
# noetig, die Whitelist ist die Regel selbst.
RUN_ID_RE = re.compile(r"^[a-z0-9-]+$")


def is_valid_run_id(run_id: str) -> bool:
    return bool(run_id) and bool(RUN_ID_RE.fullmatch(run_id))


def make_run_id(timestamp: float, *, suffix: str | None = None) -> str:
    """Zeit-sortierbare Run-ID: `<UTC-Zeitstempel><zufaelliges Suffix>`.

    UTC (statt lokaler Zeit) macht die lexikographische Sortierung
    zeitzonen-unabhaengig. Das Suffix vermeidet Kollisionen bei zwei Laeufen,
    die in derselben Sekunde enden (injizierbar fuer deterministische Tests).
    """
    stamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    suf = suffix if suffix is not None else uuid.uuid4().hex[:6]
    return f"{stamp}-{suf}"


def build_run_record(
    *,
    run_id: str,
    started_at: float,
    finished_at: float,
    source_pdf: str | None,
    dry_run: bool | None,
    options: dict | None,
    rc: int | None,
    notes: list[dict],
    duration_s: float | None = None,
    tokens: dict | None = None,
) -> dict:
    """Reiner Dict-Aufbau — kein I/O, keine Validierung (die macht der Aufrufer).

    `duration_s`/`tokens` (P5: aus dem `run_summary`-Event, s. run_parser.py)
    sind optional — fehlen sie (kein run_summary-Event im Lauf, z.B. Crash vor
    dem Final-Report), tauchen die Keys gar nicht im Record auf statt mit `None`
    aufzufuellen (kein Erfinden, L5).
    """
    record = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "source_pdf": source_pdf,
        "dry_run": dry_run,
        "options": options or {},
        "rc": rc,
        "notes": notes,
    }
    if duration_s is not None:
        record["duration_s"] = duration_s
    if tokens:
        record["tokens"] = tokens
    return record


def write_run_record(record: dict, runs_dir: Path | str) -> Path:
    """Schreibt einen Record atomar (Tempfile im selben Verzeichnis + `os.replace`)."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / f"{record['run_id']}.json"
    fd, tmp_name = tempfile.mkstemp(dir=runs_dir, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def read_run_record(run_id: str, runs_dir: Path | str) -> dict | None:
    """Liest einen Record. `None` bei ungueltiger `run_id`, fehlender oder
    kaputter Datei (geloggt, kein Crash — L5 „fail-closed und sichtbar“)."""
    if not is_valid_run_id(run_id):
        return None
    runs_root = Path(runs_dir).resolve()
    path = (runs_root / f"{run_id}.json").resolve()
    if not path.is_relative_to(runs_root) or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Kaputter Run-Record uebersprungen (%s): %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Run-Record kein Objekt, uebersprungen: %s", path)
        return None
    return data


def list_run_records(runs_dir: Path | str, *, limit: int = 50) -> list[dict]:
    """Neueste zuerst, kaputte Records werden uebersprungen (nicht gecrasht)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    records: list[dict] = []
    for f in sorted(runs_dir.glob("*.json")):
        run_id = f.stem
        if not is_valid_run_id(run_id):
            continue
        record = read_run_record(run_id, runs_dir)
        if record is not None:
            records.append(record)
    records.sort(key=lambda r: (r.get("finished_at") or 0, r.get("run_id") or ""), reverse=True)
    return records[:limit]


def prune_old_records(runs_dir: Path | str, *, keep: int = 50) -> None:
    """Loescht die aeltesten Records, wenn mehr als `keep` vorhanden sind.
    Loescht ausschliesslich Record-Dateien im runs_dir — nie Notes."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return
    entries: list[tuple[float, Path]] = []
    for f in runs_dir.glob("*.json"):
        run_id = f.stem
        if not is_valid_run_id(run_id):
            continue
        record = read_run_record(run_id, runs_dir)
        finished_at = record.get("finished_at") if record else None
        try:
            sort_key = float(finished_at) if finished_at is not None else f.stat().st_mtime
        except (TypeError, ValueError):
            sort_key = f.stat().st_mtime
        entries.append((sort_key, f))
    entries.sort(key=lambda t: t[0], reverse=True)  # neueste zuerst
    for _, f in entries[keep:]:
        try:
            f.unlink()
        except OSError as exc:
            logger.warning("Konnte alten Run-Record nicht loeschen (%s): %s", f, exc)
