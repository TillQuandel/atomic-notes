"""Integrationstests für den calibration_labels-DB-Write in collect.main().

Hintergrund (Review-Findings zum #24-Branch):
- collect.py las note_evals aus `generative/.cache/atomic_analytics.db`, die
  kanonische DB liegt aber unter `<repo>/.cache/` (db.DB_PATH) — der sqlite-Fehler
  brach den GESAMTEN Write-Block still ab (äußerer except).
- pipeline_labels == {} (AGENT_VERSION ohne quality_history-Einträge) blieb ohne
  Warnung → agreement_rate still None für alle Notes.
- Doppelte (note, claim_idx) über mehrere Label-Files: last-wins ohne Warnung.
"""
from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db
from calibration import collect
from calibration import kappa
from config import AGENT_VERSION


def _write_label_file(notes_dir: Path, name: str, rows: list[tuple[int, str]]) -> None:
    lines = ["| Label (s/h/?) | Tag | Notiz |", "|---|---|---|"]
    for idx, label in rows:
        lines.append(f"| <!--claim_idx={idx}--> {label} | | |")
    (notes_dir / name).write_text("\n".join(lines), encoding="utf-8")


def _setup(tmp_path, monkeypatch, files, sample_entries, history_lines=()):
    notes_dir = tmp_path / "labels-active"
    notes_dir.mkdir()
    for name, rows in files:
        _write_label_file(notes_dir, name, rows)
    sample_file = tmp_path / "sample.jsonl"
    sample_file.write_text(
        "\n".join(json.dumps(e) for e in sample_entries) + "\n", encoding="utf-8")
    history = tmp_path / "quality_history.jsonl"
    history.write_text("\n".join(history_lines), encoding="utf-8")
    db_file = tmp_path / "analytics.db"
    db.init_db(db_file)
    monkeypatch.setattr(collect, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(collect, "SAMPLE_FILE", sample_file)
    monkeypatch.setattr(collect, "OUTPUT", tmp_path / "labels_human.jsonl")
    monkeypatch.setattr(kappa, "HISTORY_PATH", history)
    monkeypatch.setattr(db, "DB_PATH", db_file)
    return db_file


def test_main_writes_calibration_labels_to_canonical_db(tmp_path, monkeypatch):
    db_file = _setup(
        tmp_path, monkeypatch,
        files=[("01__n.md", [(0, "s"), (1, "h")])],
        sample_entries=[{"note_name": "n.md", "language_pair": "de_de"}],
    )
    collect.main()
    conn = sqlite3.connect(str(db_file))
    rows = conn.execute(
        "SELECT note_path, n_claims, n_supported, n_hallucinated FROM calibration_labels"
    ).fetchall()
    conn.close()
    assert rows == [("n.md", 2, 1, 1)]


def test_agreement_rate_from_pipeline_labels_end_to_end(tmp_path, monkeypatch):
    history = json.dumps({
        "version": AGENT_VERSION,
        "note": "n.md",
        "claim_scores": [
            {"claim_idx": 0, "label": "supported_exact"},    # → s, Mensch: s → einig
            {"claim_idx": 1, "label": "not_in_context"},      # → h, Mensch: s → uneinig
        ],
    })
    db_file = _setup(
        tmp_path, monkeypatch,
        files=[("01__n.md", [(0, "s"), (1, "s")])],
        sample_entries=[{"note_name": "n.md", "language_pair": "de_de"}],
        history_lines=[history],
    )
    collect.main()
    conn = sqlite3.connect(str(db_file))
    (agree,) = conn.execute(
        "SELECT agreement_rate FROM calibration_labels WHERE note_path='n.md'"
    ).fetchone()
    conn.close()
    assert agree == 0.5


def test_warns_when_no_pipeline_labels_for_current_version(tmp_path, monkeypatch, capsys):
    stale = json.dumps({
        "version": "v0.0.1-stale",
        "note": "n.md",
        "claim_scores": [{"claim_idx": 0, "label": "supported_exact"}],
    })
    _setup(
        tmp_path, monkeypatch,
        files=[("01__n.md", [(0, "s")])],
        sample_entries=[{"note_name": "n.md", "language_pair": "de_de"}],
        history_lines=[stale],
    )
    collect.main()
    err = capsys.readouterr().err
    assert "0 Pipeline-Labels" in err
    assert AGENT_VERSION in err


def test_conflicting_duplicate_labels_warn(tmp_path, monkeypatch, capsys):
    # Zwei Label-Files mappen auf dieselbe note_name und widersprechen sich bei
    # claim 0 — per_note summiert beide, human_by_note behält still das letzte.
    # Mindestens eine Warnung muss den Konflikt sichtbar machen.
    _setup(
        tmp_path, monkeypatch,
        files=[("01__a.md", [(0, "s")]), ("02__b.md", [(0, "h")])],
        sample_entries=[
            {"note_name": "n.md", "language_pair": "de_de"},
            {"note_name": "n.md", "language_pair": "en_de"},
        ],
    )
    collect.main()
    err = capsys.readouterr().err
    assert "widersprüchliche Labels" in err
    assert "n.md" in err
