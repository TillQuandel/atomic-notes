"""Tests fuer `--repeat` (Wiederholungs-Sweeps, reeval_baseline.py).

Kontext: die Trendfaehigkeits-Studie (2026-07-16) hat belegt, dass bisherige
"Wiederholungen" von reeval_baseline.py-Laeufen keine echten Wiederholungs-
messungen sind -- der content-adressierte Judge-Cache (EVAL_CACHE_NAMESPACE,
s. eval_quality_v4.py `_call_judge`) liefert fuer dieselbe Note byte-
identische Ergebnisse, auch aus einem neuen Prozess/Tag heraus. `--repeat`
schafft die Voraussetzung fuer echtes Eval-Rauschen und wirkt additiv:

(a) Der `_already_done`-Resume-Skip wird ignoriert -- eine Note mit bereits
    vorhandener aktueller-EVAL_VERSION-Zeile wird TROTZDEM erneut evaluiert.
    Nichts wird ueberschrieben (mehrere Zeilen je Note+eval_version sind im
    Schema zulaessig).
(b) `eval_note(..., no_cache=True)` erzwingt eine echte Judge-Neuberechnung.
    Mechanik-Analyse (Codeanker, kein Raten): reeval_baseline.py durchlaeuft
    genau EINE Wiederverwendungs-Ebene -- den content-adressierten LLM-Call-
    Cache in `_call_judge`/`call_claude_full` (eval_quality_v4.py Z. 1111,
    1121; agents/base.py `_cache_get`/`_cache_put`), gesteuert ueber
    `eval_note(no_cache=...)`. Der zweite bekannte Mechanismus, der Re-Eval-
    Hash-Guard `find_cached_eval` (eval_quality_v4.py Z. 1198), wird
    ausschliesslich von `orchestrator.py::run_stage8_eval` aufgerufen (mit
    einem echten content_hash aus dem Note-Frontmatter, Z. 1531) --
    reeval_baseline.py uebergibt `eval_note()` nie einen content_hash und
    ruft `find_cached_eval()` nirgends auf. Dieser Guard greift hier also
    strukturell nicht und muss folglich auch nicht extra umgangen werden
    (siehe PR-Body fuer die volle Analyse). Retrieval/Embeddings-Caches
    (`_PDF_ARTIFACTS_CACHE`, `_CHUNK_EMB_CACHE`) sind reiner Prozess-Speicher
    ohne Disk-Persistenz -- jeder `--repeat`-Sweep ist ein neuer Prozess,
    Retrieval wird also ohnehin bei jedem Lauf frisch berechnet.
(c) Neue Zeilen werden mit `repeat_sweep: true` markiert (quality_history.
    jsonl-Eintrag); der Run wird in pipeline_runs mit
    pipeline_version='reeval-repeat' (statt 'reeval') registriert, damit die
    Rauschanalyse Wiederholungsgruppen sicher findet.

Ohne --repeat bleibt das Verhalten byte-identisch (Regressionstests je Block).
"""

from __future__ import annotations

from pathlib import Path

from generative import db as _db
from generative import reeval_baseline


def _make_baseline(tmp_path: Path, monkeypatch) -> Path:
    """Ein Baseline-Ordner mit genau einer Note + gemapptem Fake-PDF."""
    baseline_root = tmp_path / "baseline"
    (baseline_root / "Foo").mkdir(parents=True)
    (baseline_root / "Foo" / "vault__X.md").write_text("x", encoding="utf-8")
    fake_pdf = tmp_path / "Foo.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(reeval_baseline, "PDF_MAP", {"Foo": fake_pdf})
    return baseline_root


def _wire_test_db(tmp_path: Path, monkeypatch) -> Path:
    """Isolierte Test-DB statt der echten atomic_analytics.db (wie TestBaselineDirOption)."""
    test_db_path = tmp_path / "test.db"
    _db.init_db(test_db_path)
    original_get_db = _db.get_db  # vor dem Patch sichern, sonst rekursiver Selbstaufruf
    monkeypatch.setattr(reeval_baseline._db, "get_db", lambda path=None: original_get_db(test_db_path))
    return test_db_path


def _mark_done_at_current_eval_version(test_db_path: Path, note_name: str = "vault__X.md") -> None:
    with _db.get_db(test_db_path) as conn:
        _db.insert_run(conn, {"run_id": "r0", "pipeline_version": "v0"})
        _db.insert_eval(
            conn,
            {
                "eval_id": f"r0__{note_name}",
                "run_id": "r0",
                "note_path": note_name,
                "eval_version": reeval_baseline._eq.EVAL_VERSION,
            },
        )


def _fake_eval_note_factory(calls: list[dict]):
    """Stub fuer `_eq.eval_note` -- zeichnet insb. `no_cache` auf, ohne echten
    Judge-/PDF-/Retrieval-Aufwand (kein LLM-Call, keine echten Notes/PDFs noetig)."""

    def _inner(note_path, pdf_path, pipeline_version=None, no_cache=False, content_hash=None, **kwargs):
        calls.append({"note_path": note_path, "no_cache": no_cache})
        return {
            "note": note_path.name,
            "pdf": pdf_path.name,
            "version": pipeline_version,
            "eval_version": reeval_baseline._eq.EVAL_VERSION,
            "content_hash": content_hash,
            "hallucination_rate": 0.5,
            "coverage_factual": 0.5,
            "coverage_rate": 0.5,
            "claims_total": 1,
        }

    return _inner


# ---------------------------------------------------------------------------
# (a) --repeat deaktiviert den _already_done-Resume-Skip (additiv)
# ---------------------------------------------------------------------------


class TestRepeatBypassesAlreadyDoneSkip:
    def test_without_repeat_note_with_current_version_is_skipped(self, tmp_path, monkeypatch, capsys):
        """Regression: unveraendertes Verhalten ohne --repeat."""
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        test_db_path = _wire_test_db(tmp_path, monkeypatch)
        _mark_done_at_current_eval_version(test_db_path)

        reeval_baseline.main(["--dry-run", "--baseline-dir", str(baseline_root)])

        out = capsys.readouterr().out
        assert "skip (bereits" in out
        assert "Fertig: 0 neu evaluiert, 1 übersprungen, 0 Fehler" in out

    def test_with_repeat_note_with_current_version_is_not_skipped(self, tmp_path, monkeypatch, capsys):
        """RED auf altem Code: --repeat existiert nicht -> argparse bricht ab."""
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        test_db_path = _wire_test_db(tmp_path, monkeypatch)
        _mark_done_at_current_eval_version(test_db_path)

        reeval_baseline.main(["--dry-run", "--repeat", "--baseline-dir", str(baseline_root)])

        out = capsys.readouterr().out
        assert "skip (bereits" not in out
        assert "Fertig: 1 neu evaluiert, 0 übersprungen, 0 Fehler" in out

    def test_repeat_does_not_affect_notes_without_existing_eval(self, tmp_path, monkeypatch, capsys):
        """Notes ohne vorhandene Zeile verhalten sich mit/ohne --repeat identisch
        (kein Skip in beiden Faellen -- --repeat aendert nur den Skip-Fall)."""
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        _wire_test_db(tmp_path, monkeypatch)  # leere DB, keine vorhandene Zeile

        reeval_baseline.main(["--dry-run", "--repeat", "--baseline-dir", str(baseline_root)])

        out = capsys.readouterr().out
        assert "Fertig: 1 neu evaluiert, 0 übersprungen, 0 Fehler" in out


# ---------------------------------------------------------------------------
# (b) --repeat erzwingt echte Judge-Neuberechnung (no_cache durchgereicht)
# ---------------------------------------------------------------------------


class TestRepeatForcesFreshEvaluation:
    def test_without_repeat_eval_note_called_with_no_cache_false(self, tmp_path, monkeypatch):
        """Regression: Default-Verhalten unveraendert (no_cache=False wie bisher
        implizit, Judge-Cache aktiv)."""
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        _wire_test_db(tmp_path, monkeypatch)

        calls: list[dict] = []
        monkeypatch.setattr(reeval_baseline._eq, "eval_note", _fake_eval_note_factory(calls))
        save_calls: list[dict] = []
        monkeypatch.setattr(reeval_baseline._eq, "save_result", lambda result: save_calls.append(result))

        reeval_baseline.main(["--baseline-dir", str(baseline_root)])

        assert len(calls) == 1
        assert calls[0]["no_cache"] is False

    def test_with_repeat_eval_note_called_with_no_cache_true(self, tmp_path, monkeypatch):
        """--repeat muss no_cache=True durchreichen -- das ist die einzige Ebene,
        die reeval_baseline.py ueberhaupt durchlaeuft (s. Moduldocstring)."""
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        _wire_test_db(tmp_path, monkeypatch)

        calls: list[dict] = []
        monkeypatch.setattr(reeval_baseline._eq, "eval_note", _fake_eval_note_factory(calls))
        save_calls: list[dict] = []
        monkeypatch.setattr(reeval_baseline._eq, "save_result", lambda result: save_calls.append(result))

        reeval_baseline.main(["--repeat", "--baseline-dir", str(baseline_root)])

        assert len(calls) == 1
        assert calls[0]["no_cache"] is True


# ---------------------------------------------------------------------------
# (c) Kennzeichnung: repeat_sweep-Feld + pipeline_runs-Label 'reeval-repeat'
# ---------------------------------------------------------------------------


class TestRepeatMarking:
    def test_without_repeat_result_has_no_repeat_sweep_field(self, tmp_path, monkeypatch):
        """Regression: additives Feld darf ohne --repeat gar nicht erst auftauchen
        (sonst waere quality_history.jsonl nicht byte-identisch zu heute)."""
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        _wire_test_db(tmp_path, monkeypatch)

        monkeypatch.setattr(reeval_baseline._eq, "eval_note", _fake_eval_note_factory([]))
        save_calls: list[dict] = []
        monkeypatch.setattr(reeval_baseline._eq, "save_result", lambda result: save_calls.append(result))

        reeval_baseline.main(["--baseline-dir", str(baseline_root)])

        assert len(save_calls) == 1
        assert "repeat_sweep" not in save_calls[0]

    def test_with_repeat_result_marked_repeat_sweep_true(self, tmp_path, monkeypatch):
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        _wire_test_db(tmp_path, monkeypatch)

        monkeypatch.setattr(reeval_baseline._eq, "eval_note", _fake_eval_note_factory([]))
        save_calls: list[dict] = []
        monkeypatch.setattr(reeval_baseline._eq, "save_result", lambda result: save_calls.append(result))

        reeval_baseline.main(["--repeat", "--baseline-dir", str(baseline_root)])

        assert len(save_calls) == 1
        assert save_calls[0]["repeat_sweep"] is True

    def test_without_repeat_pipeline_runs_labeled_reeval(self, tmp_path, monkeypatch):
        """Regression: bestehendes Label bleibt unveraendert."""
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        test_db_path = _wire_test_db(tmp_path, monkeypatch)
        monkeypatch.setattr(reeval_baseline._eq, "eval_note", _fake_eval_note_factory([]))
        monkeypatch.setattr(reeval_baseline._eq, "save_result", lambda result: None)

        reeval_baseline.main(["--baseline-dir", str(baseline_root)])

        runs = _db.query_pipeline_runs(test_db_path)
        assert len(runs) == 1
        assert runs[0]["pipeline_version"] == "reeval"

    def test_with_repeat_pipeline_runs_labeled_reeval_repeat(self, tmp_path, monkeypatch):
        baseline_root = _make_baseline(tmp_path, monkeypatch)
        test_db_path = _wire_test_db(tmp_path, monkeypatch)
        monkeypatch.setattr(reeval_baseline._eq, "eval_note", _fake_eval_note_factory([]))
        monkeypatch.setattr(reeval_baseline._eq, "save_result", lambda result: None)

        reeval_baseline.main(["--repeat", "--baseline-dir", str(baseline_root)])

        runs = _db.query_pipeline_runs(test_db_path)
        assert len(runs) == 1
        assert runs[0]["pipeline_version"] == "reeval-repeat"


# ---------------------------------------------------------------------------
# Sicherheit: --repeat kombinierbar mit --dry-run (zeigt, was liefe)
# ---------------------------------------------------------------------------


def test_repeat_and_dry_run_combinable_no_eval_note_call(tmp_path, monkeypatch, capsys):
    baseline_root = _make_baseline(tmp_path, monkeypatch)
    test_db_path = _wire_test_db(tmp_path, monkeypatch)
    _mark_done_at_current_eval_version(test_db_path)

    calls: list[dict] = []
    monkeypatch.setattr(reeval_baseline._eq, "eval_note", _fake_eval_note_factory(calls))

    reeval_baseline.main(["--dry-run", "--repeat", "--baseline-dir", str(baseline_root)])

    assert calls == []  # dry-run ruft eval_note nie auf, auch nicht mit --repeat
    out = capsys.readouterr().out
    assert "[dry-run]" in out
