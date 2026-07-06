"""Tests fuer den Re-Eval-Hash-Guard (Stage 8).

Eine Note, deren Inhalt (pipeline-content-hash aus dem Frontmatter, #47) und deren
eval_version + pipeline_version bereits in quality_history.jsonl mit demselben
Ergebnis stehen, wird nicht erneut vom Judge bewertet — das gesparte Ergebnis wird
fuer die Lauf-Aggregation wiederverwendet. `--fresh-run` erzwingt trotzdem Re-Eval.

Diese Datei prueft zwei Ebenen:
1. `eval_quality_v4.find_cached_eval` — die reine Match-Logik gegen ein
   quality_history.jsonl (Hash + Version-Scoping, Backward-Compat).
2. `orchestrator.run_stage8_eval` — die Stage-8-Schleife, die den Guard vor jedem
   Judge-Call anwendet, sichtbar zusammenfasst und `--fresh-run` bypassed.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from generative import eval_quality_v4 as eq
from generative import orchestrator
from generative.pipeline import vault_writer


def _note_text(body: str = "Body unveraendert.\n") -> str:
    rendered = f'---\ntitle: "Test Note"\n---\n{body}'
    return vault_writer.inject_content_hash(rendered)


NOTE_TEXT = _note_text()
NOTE_HASH = vault_writer.extract_content_hash(NOTE_TEXT)
assert NOTE_HASH  # Testvoraussetzung: inject_content_hash hat tatsaechlich gehasht.


def _write_history(path: Path, *entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _base_entry(**overrides) -> dict:
    entry = {
        "note": "test-note.md",
        "pdf": "quelle.pdf",
        "version": orchestrator.AGENT_VERSION,
        "eval_version": eq.EVAL_VERSION,
        "content_hash": NOTE_HASH,
        "hallucination_rate": 0.11,
        "coverage_rate": 0.8,
        "claims_total": 5,
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# Ebene 1: reine Match-Logik
# ---------------------------------------------------------------------------


class TestFindCachedEval:
    def test_hit_on_matching_hash_and_versions(self, tmp_path):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry())

        result = eq.find_cached_eval(NOTE_HASH, eq.EVAL_VERSION, orchestrator.AGENT_VERSION, history_path=history)

        assert result is not None
        assert result["content_hash"] == NOTE_HASH

    def test_no_hit_on_hash_mismatch(self, tmp_path):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry(content_hash="deadbeef"))

        result = eq.find_cached_eval(NOTE_HASH, eq.EVAL_VERSION, orchestrator.AGENT_VERSION, history_path=history)

        assert result is None

    def test_no_hit_on_eval_version_mismatch(self, tmp_path):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry(eval_version="3.9"))

        result = eq.find_cached_eval(NOTE_HASH, eq.EVAL_VERSION, orchestrator.AGENT_VERSION, history_path=history)

        assert result is None

    def test_no_hit_on_pipeline_version_mismatch(self, tmp_path):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry(version="v0.0.1-andere-version"))

        result = eq.find_cached_eval(NOTE_HASH, eq.EVAL_VERSION, orchestrator.AGENT_VERSION, history_path=history)

        assert result is None

    def test_no_hit_on_missing_content_hash_field(self, tmp_path):
        """Backward-Compat: Alt-Records ohne content_hash-Feld matchen nie."""
        history = tmp_path / "quality_history.jsonl"
        entry = _base_entry()
        del entry["content_hash"]
        _write_history(history, entry)

        result = eq.find_cached_eval(NOTE_HASH, eq.EVAL_VERSION, orchestrator.AGENT_VERSION, history_path=history)

        assert result is None

    def test_no_history_file_returns_none(self, tmp_path):
        result = eq.find_cached_eval(
            NOTE_HASH, eq.EVAL_VERSION, orchestrator.AGENT_VERSION, history_path=tmp_path / "missing.jsonl"
        )
        assert result is None

    def test_empty_hash_returns_none(self, tmp_path):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry())

        result = eq.find_cached_eval(None, eq.EVAL_VERSION, orchestrator.AGENT_VERSION, history_path=history)

        assert result is None

    def test_latest_matching_entry_wins(self, tmp_path):
        history = tmp_path / "quality_history.jsonl"
        _write_history(
            history,
            _base_entry(hallucination_rate=0.11),
            _base_entry(hallucination_rate=0.22),
        )

        result = eq.find_cached_eval(NOTE_HASH, eq.EVAL_VERSION, orchestrator.AGENT_VERSION, history_path=history)

        assert result["hallucination_rate"] == 0.22


class TestEvalNoteThreadsContentHash:
    def test_empty_result_includes_content_hash(self):
        result = eq._empty_result(
            Path("missing-note.md"),
            Path("missing.pdf"),
            "v0.3.140",
            "2026-01-01T00:00:00",
            "note_not_found",
            content_hash="abc123",
        )
        assert result["content_hash"] == "abc123"

    def test_eval_note_missing_note_threads_content_hash(self, tmp_path):
        result = eq.eval_note(tmp_path / "missing.md", tmp_path / "missing.pdf", content_hash="abc123")
        assert result["error"] == "note_not_found"
        assert result["content_hash"] == "abc123"

    def test_eval_note_default_content_hash_is_none(self, tmp_path):
        result = eq.eval_note(tmp_path / "missing.md", tmp_path / "missing.pdf")
        assert result["content_hash"] is None


# ---------------------------------------------------------------------------
# Ebene 2: Stage-8-Schleife im Orchestrator
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _fake_get_db(path=None):
    yield None


def _write_note(tmp_path: Path, name: str = "note.md", body: str = "Body unveraendert.\n") -> Path:
    note_path = tmp_path / name
    note_path.write_text(_note_text(body), encoding="utf-8")
    return note_path


def _fake_eval_note(calls: list[dict]):
    def _inner(note_path, pdf_path, pipeline_version=None, content_hash=None, **kwargs):
        calls.append({"note_path": note_path, "content_hash": content_hash})
        return {
            "note": note_path.name,
            "pdf": pdf_path.name,
            "version": pipeline_version,
            "eval_version": eq.EVAL_VERSION,
            "content_hash": content_hash,
            "hallucination_rate": 0.99,  # Marker: klar unterscheidbar vom Alt-Record.
            "coverage_rate": 0.5,
            "claims_total": 3,
        }

    return _inner


class TestRunStage8Eval:
    def test_cache_hit_skips_judge_and_reuses_old_result(self, tmp_path, monkeypatch, capsys):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry(hallucination_rate=0.11))
        monkeypatch.setattr(eq, "_QUALITY_HISTORY", history)

        judge_calls: list[dict] = []
        monkeypatch.setattr(eq, "eval_note", _fake_eval_note(judge_calls))
        save_calls: list[dict] = []
        monkeypatch.setattr(eq, "save_result", lambda result: save_calls.append(result))

        note_path = _write_note(tmp_path)
        pdf_path = tmp_path / "quelle.pdf"

        eval_results, evaluated_count, reused_count = orchestrator.run_stage8_eval(
            [note_path], pdf_path, {}, fresh_run=False
        )

        assert judge_calls == []  # Judge NICHT gerufen.
        assert save_calls == []  # kein neuer Record noetig.
        assert evaluated_count == 0
        assert reused_count == 1
        assert len(eval_results) == 1
        assert eval_results[0]["hallucination_rate"] == 0.11  # stammt aus Alt-Record, nicht aus Fake-Judge (0.99).

        out = capsys.readouterr().out
        assert "[8/8]" in out
        assert "0 evaluiert" in out
        assert "1 unveraendert uebernommen" in out
        assert "Hash-Guard" in out

    def test_hash_mismatch_runs_judge(self, tmp_path, monkeypatch):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry(content_hash="andere-note-hash"))
        monkeypatch.setattr(eq, "_QUALITY_HISTORY", history)

        judge_calls: list[dict] = []
        monkeypatch.setattr(eq, "eval_note", _fake_eval_note(judge_calls))
        save_calls: list[dict] = []
        monkeypatch.setattr(eq, "save_result", lambda result: save_calls.append(result))

        note_path = _write_note(tmp_path)
        pdf_path = tmp_path / "quelle.pdf"

        eval_results, evaluated_count, reused_count = orchestrator.run_stage8_eval(
            [note_path], pdf_path, {}, fresh_run=False
        )

        assert len(judge_calls) == 1
        assert judge_calls[0]["content_hash"] == NOTE_HASH
        assert evaluated_count == 1
        assert reused_count == 0
        assert len(save_calls) == 1
        assert eval_results[0]["hallucination_rate"] == 0.99

    def test_eval_version_mismatch_runs_judge(self, tmp_path, monkeypatch):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry(eval_version="3.9"))
        monkeypatch.setattr(eq, "_QUALITY_HISTORY", history)

        judge_calls: list[dict] = []
        monkeypatch.setattr(eq, "eval_note", _fake_eval_note(judge_calls))
        monkeypatch.setattr(eq, "save_result", lambda result: None)

        note_path = _write_note(tmp_path)
        pdf_path = tmp_path / "quelle.pdf"

        _, evaluated_count, reused_count = orchestrator.run_stage8_eval([note_path], pdf_path, {}, fresh_run=False)

        assert len(judge_calls) == 1
        assert evaluated_count == 1
        assert reused_count == 0

    def test_fresh_run_forces_reeval_despite_identical_hash(self, tmp_path, monkeypatch):
        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry())  # exakter Treffer waere moeglich
        monkeypatch.setattr(eq, "_QUALITY_HISTORY", history)

        judge_calls: list[dict] = []
        monkeypatch.setattr(eq, "eval_note", _fake_eval_note(judge_calls))
        monkeypatch.setattr(eq, "save_result", lambda result: None)

        note_path = _write_note(tmp_path)
        pdf_path = tmp_path / "quelle.pdf"

        _, evaluated_count, reused_count = orchestrator.run_stage8_eval([note_path], pdf_path, {}, fresh_run=True)

        assert len(judge_calls) == 1  # --fresh-run bypassed den Guard trotz Hash-Treffer.
        assert evaluated_count == 1
        assert reused_count == 0

    def test_legacy_record_without_hash_field_runs_judge_and_new_record_gets_field(self, tmp_path, monkeypatch):
        history = tmp_path / "quality_history.jsonl"
        legacy_entry = _base_entry()
        del legacy_entry["content_hash"]  # Alt-Record vor diesem Feature.
        _write_history(history, legacy_entry)
        monkeypatch.setattr(eq, "_QUALITY_HISTORY", history)

        judge_calls: list[dict] = []
        monkeypatch.setattr(eq, "eval_note", _fake_eval_note(judge_calls))
        save_calls: list[dict] = []
        monkeypatch.setattr(eq, "save_result", lambda result: save_calls.append(result))

        note_path = _write_note(tmp_path)
        pdf_path = tmp_path / "quelle.pdf"

        _, evaluated_count, reused_count = orchestrator.run_stage8_eval([note_path], pdf_path, {}, fresh_run=False)

        assert len(judge_calls) == 1
        assert evaluated_count == 1
        assert reused_count == 0
        assert len(save_calls) == 1
        assert save_calls[0]["content_hash"] == NOTE_HASH  # neuer Record traegt das Feld.

    def test_aggregation_counts_mixed_hit_and_miss_correctly(self, tmp_path, monkeypatch, capsys):
        hit_note = _write_note(tmp_path, "hit.md", "Unveraenderter Inhalt.\n")
        hit_hash = vault_writer.extract_content_hash(hit_note.read_text(encoding="utf-8"))
        miss_note = _write_note(tmp_path, "miss.md", "Ein komplett anderer Inhalt.\n")

        history = tmp_path / "quality_history.jsonl"
        _write_history(history, _base_entry(note="hit.md", content_hash=hit_hash))
        monkeypatch.setattr(eq, "_QUALITY_HISTORY", history)

        judge_calls: list[dict] = []
        monkeypatch.setattr(eq, "eval_note", _fake_eval_note(judge_calls))
        save_calls: list[dict] = []
        monkeypatch.setattr(eq, "save_result", lambda result: save_calls.append(result))

        pdf_path = tmp_path / "quelle.pdf"
        eval_results, evaluated_count, reused_count = orchestrator.run_stage8_eval(
            [hit_note, miss_note], pdf_path, {}, fresh_run=False
        )

        # keine doppelten/fehlenden Notes in den Kennzahlen:
        assert len(eval_results) == 2
        assert evaluated_count == 1
        assert reused_count == 1
        assert len(judge_calls) == 1
        assert len(save_calls) == 1

        out = capsys.readouterr().out
        assert "1 evaluiert" in out
        assert "1 unveraendert uebernommen" in out

    def test_unreadable_note_falls_back_to_normal_eval(self, tmp_path, monkeypatch):
        """Note existiert nicht (z.B. Race) -> kein Hash, Guard greift nie, Judge laeuft normal."""
        history = tmp_path / "quality_history.jsonl"
        monkeypatch.setattr(eq, "_QUALITY_HISTORY", history)

        judge_calls: list[dict] = []
        monkeypatch.setattr(eq, "eval_note", _fake_eval_note(judge_calls))
        monkeypatch.setattr(eq, "save_result", lambda result: None)

        missing_note = tmp_path / "does-not-exist.md"
        pdf_path = tmp_path / "quelle.pdf"

        _, evaluated_count, reused_count = orchestrator.run_stage8_eval([missing_note], pdf_path, {}, fresh_run=False)

        assert len(judge_calls) == 1
        assert judge_calls[0]["content_hash"] is None
        assert evaluated_count == 1
        assert reused_count == 0
