# -*- coding: utf-8 -*-
"""Sichtbare Meldung fuer den Stage-8-Eval-Cap (#151, Punkt 5b).

run_stage8_eval evaluiert hoechstens note_files[:10]. Ab der 11. Note wurde bisher
stumm nichts evaluiert. Greift der Cap, muss jetzt eine sichtbare Log-Zeile erscheinen.
"""

from __future__ import annotations

from pathlib import Path

from generative import eval_quality_v4 as eq
from generative import orchestrator
from generative.pipeline import vault_writer


def _note_text(body: str) -> str:
    return vault_writer.inject_content_hash(f'---\ntitle: "T"\n---\n{body}')


def _fake_eval_note(note_path, pdf_path, pipeline_version=None, content_hash=None, **kwargs):
    return {
        "note": note_path.name,
        "pdf": pdf_path.name,
        "version": pipeline_version,
        "eval_version": eq.EVAL_VERSION,
        "content_hash": content_hash,
        "hallucination_rate": 0.1,
        "coverage_rate": 0.8,
        "claims_total": 3,
    }


def _make_notes(tmp_path: Path, count: int) -> list[Path]:
    notes = []
    for i in range(count):
        p = tmp_path / f"note-{i}.md"
        p.write_text(_note_text(f"Inhalt Nummer {i}.\n"), encoding="utf-8")
        notes.append(p)
    return notes


def test_cap_line_printed_when_more_than_ten(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(eq, "_QUALITY_HISTORY", tmp_path / "quality_history.jsonl")
    monkeypatch.setattr(eq, "eval_note", _fake_eval_note)
    monkeypatch.setattr(eq, "save_result", lambda result: None)

    notes = _make_notes(tmp_path, 13)
    _, evaluated_count, _ = orchestrator.run_stage8_eval(notes, tmp_path / "q.pdf", {}, fresh_run=False)

    out = capsys.readouterr().out
    assert "evaluiere 10 von 13 Notes (Cap)" in out
    assert evaluated_count == 10  # nur die ersten 10


def test_no_cap_line_when_ten_or_fewer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(eq, "_QUALITY_HISTORY", tmp_path / "quality_history.jsonl")
    monkeypatch.setattr(eq, "eval_note", _fake_eval_note)
    monkeypatch.setattr(eq, "save_result", lambda result: None)

    notes = _make_notes(tmp_path, 10)
    _, evaluated_count, _ = orchestrator.run_stage8_eval(notes, tmp_path / "q.pdf", {}, fresh_run=False)

    out = capsys.readouterr().out
    assert "(Cap)" not in out
    assert evaluated_count == 10
