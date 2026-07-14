"""Tests fuer #241: Dry-Run-Baseline-Ablage muss nach run_id namespacen.

Befund: Output-Notes wurden nach PDF-Stem ohne run_id abgelegt — zwei Laeufe auf
demselben PDF ueberschreiben sich gegenseitig. Realer Schaden: in der A/B-
Effizienzreihe 2026-07-13 wurden M1-Notes von M2 ueberschrieben (nur durch
Hand-Snapshots vermieden).

Fix: `.../eval/baseline/<stem>/` -> `.../eval/baseline/<stem>/<run_id>/`, Writer
(vault_writer.write_note) UND Reader (orchestrator cache_note_dir, Stage 8)
sowie die beiden nachgelagerten Konsumenten (reeval_baseline.py,
calibration/sample.py) im Lockstep.

RED auf master:
- TestWriterRunNamespace: write_note() kennt kein `run_id`-Kwarg -> TypeError.
- TestWriterReaderConsistency: dito (write_note-Aufruf mit run_id schlaegt fehl).
- TestReevalBaselineFindsNestedNotes / TestCalibrationSampleFindsNestedNotes:
  `_latest_notes_dir` existiert auf master nicht (ImportError) bzw.
  `candidate_notes` findet 0 Notes im genesteten Layout (flaches glob).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from generative.orchestrator import dry_run_eval_targets
from generative.pipeline import vault_writer
from generative.schemas.atomic_note import AtomicNoteDraft

REPO_ROOT = Path(__file__).resolve().parents[2]


def _draft(title: str = "Sample Concept", body: str = "pipeline body") -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title=title,
        body=body,
        source_anchors=[],
        related=[],
        tags=["t"],
        synthesis_confidence="low",
        critic_score=5,
        hard_gates_pass=True,
    )


class TestWriterRunNamespace:
    """Zwei "Laeufe" (verschiedene run_ids) auf demselben PDF-Stem duerfen sich
    nicht gegenseitig ueberschreiben — die eigentliche Regression aus #241."""

    def test_two_runs_same_pdf_write_separate_eval_copies(self, tmp_path):
        stem = "test-241-two-runs-fixture"
        source_file = f"{stem}.pdf"
        root = REPO_ROOT / "generative" / ".cache" / "eval" / "baseline" / stem
        try:
            vault_writer.write_note(
                _draft(body="RUN 1 BODY"),
                source_file=source_file,
                dry_run=True,
                inbox_dir=tmp_path,
                run_id="run-1",
            )
            vault_writer.write_note(
                _draft(body="RUN 2 BODY"),
                source_file=source_file,
                dry_run=True,
                inbox_dir=tmp_path,
                run_id="run-2",
            )

            run1_files = list((root / "run-1").glob("*.md"))
            run2_files = list((root / "run-2").glob("*.md"))
            assert run1_files, f"Run-1-Cache-Kopie fehlt unter {root / 'run-1'}"
            assert run2_files, f"Run-2-Cache-Kopie fehlt unter {root / 'run-2'}"
            assert "RUN 1 BODY" in run1_files[0].read_text(encoding="utf-8")
            assert "RUN 2 BODY" in run2_files[0].read_text(encoding="utf-8")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_id_defaults_to_generated_namespace_when_omitted(self, tmp_path):
        """Ohne explizites run_id generiert write_note selbst einen Zeitstempel-
        Namespace (kein Import aus `generative.agents` — Schichtungs-Test #153
        verbietet Privat-Importe ueber die agents<->pipeline-Grenze). In
        Produktion (orchestrator.py, ausserhalb dieser Grenze) wird run_id
        deshalb IMMER explizit uebergeben (derselbe `_RUN_ID` wie beim Reader);
        dieser Test deckt nur den Fallback-Pfad ab (Aufrufer ohne run_id-Bedarf,
        z.B. bestehende Tests wie test_overwrite_diff.py)."""
        stem = "test-241-default-run-id-fixture"
        source_file = f"{stem}.pdf"
        root = REPO_ROOT / "generative" / ".cache" / "eval" / "baseline" / stem
        try:
            vault_writer.write_note(
                _draft(body="DEFAULT RUN ID BODY"),
                source_file=source_file,
                dry_run=True,
                inbox_dir=tmp_path,
            )
            run_subdirs = [d for d in root.iterdir() if d.is_dir()]
            assert len(run_subdirs) == 1, f"Erwartet genau einen run_id-Unterordner, gefunden: {run_subdirs}"
            cache_files = list(run_subdirs[0].glob("*.md"))
            assert cache_files, f"Cache-Kopie fehlt unter {run_subdirs[0]}"
            assert "DEFAULT RUN ID BODY" in cache_files[0].read_text(encoding="utf-8")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestWriterReaderConsistency:
    """Reader (orchestrator.dry_run_eval_targets) muss die vom Writer im selben
    run_id-Namespace geschriebenen Notes finden."""

    def test_reader_finds_notes_writer_wrote_under_same_run_id(self, tmp_path):
        stem = "test-241-reader-consistency-fixture"
        source_file = f"{stem}.pdf"
        run_id = "run-consistency-1"
        root = REPO_ROOT / "generative" / ".cache" / "eval" / "baseline" / stem
        try:
            draft = _draft(body="READER CONSISTENCY BODY")
            target = vault_writer.write_note(
                draft,
                source_file=source_file,
                dry_run=True,
                inbox_dir=tmp_path,
                run_id=run_id,
            )
            will_vault, _ = vault_writer.auto_write_decision(draft)
            assert will_vault, "Testvoraussetzung: Draft muss vault-tauglich sein (sonst prefix != 'vault')"

            cache_note_dir = root / run_id
            found = dry_run_eval_targets([(target, will_vault)], cache_note_dir)

            assert found == [cache_note_dir / f"vault__{target.name}"]
            assert found[0].exists()
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestReevalBaselineFindsNestedNotes:
    """reeval_baseline.py muss Notes finden, die unter <stem>/<run_id>/ liegen
    (statt flach unter <stem>/). Bei mehreren run_ids gilt die neueste als
    gueltige Baseline — reeval_baseline re-evaluiert den zuletzt geschriebenen
    Stand, keine veralteten Zwischenlaeufe."""

    def test_picks_newest_run_id_subdir(self, tmp_path):
        from generative.reeval_baseline import _latest_notes_dir

        pdf_dir = tmp_path / "Some-Stem"
        pdf_dir.mkdir()
        (pdf_dir / "20260101-000000").mkdir()
        (pdf_dir / "20260101-000000" / "vault__Old.md").write_text("old", encoding="utf-8")
        (pdf_dir / "20260113-120000").mkdir()
        (pdf_dir / "20260113-120000" / "vault__New.md").write_text("new", encoding="utf-8")

        notes_dir = _latest_notes_dir(pdf_dir)

        assert notes_dir.name == "20260113-120000"
        assert sorted(notes_dir.glob("vault__*.md")) == [notes_dir / "vault__New.md"]

    def test_legacy_flat_layout_without_run_id_subdir_still_works(self, tmp_path):
        from generative.reeval_baseline import _latest_notes_dir

        pdf_dir = tmp_path / "Legacy-Stem"
        pdf_dir.mkdir()
        (pdf_dir / "vault__Legacy.md").write_text("legacy", encoding="utf-8")

        notes_dir = _latest_notes_dir(pdf_dir)

        assert notes_dir == pdf_dir
        assert list(notes_dir.glob("vault__*.md")) == [pdf_dir / "vault__Legacy.md"]


class TestCalibrationSampleFindsNestedNotes:
    """calibration/sample.py's candidate_notes() muss Notes unter
    <stem>/<run_id>/ finden statt nur flach unter <stem>/. Dieselbe
    Dedup-Entscheidung wie reeval_baseline: neueste run_id gewinnt (sonst
    waere ein mehrfach prozessiertes PDF im Kalibrierungs-Sample
    ueberrepraesentiert)."""

    def test_candidate_notes_finds_notes_nested_under_run_id(self, tmp_path):
        from generative.calibration.sample import candidate_notes

        folder = tmp_path / "Some-Stem"
        folder.mkdir()
        run_dir = folder / "20260113-120000"
        run_dir.mkdir()
        (run_dir / "vault__New.md").write_text("new", encoding="utf-8")
        (run_dir / "inbox__New2.md").write_text("new2", encoding="utf-8")

        notes = candidate_notes(folder)

        assert notes == sorted([run_dir / "vault__New.md", run_dir / "inbox__New2.md"])

    def test_candidate_notes_picks_newest_run_id_subdir(self, tmp_path):
        from generative.calibration.sample import candidate_notes

        folder = tmp_path / "Some-Stem"
        folder.mkdir()
        (folder / "20260101-000000").mkdir()
        (folder / "20260101-000000" / "vault__Old.md").write_text("old", encoding="utf-8")
        newest = folder / "20260113-120000"
        newest.mkdir()
        (newest / "vault__New.md").write_text("new", encoding="utf-8")

        notes = candidate_notes(folder)

        assert notes == [newest / "vault__New.md"]

    def test_legacy_flat_layout_without_run_id_subdir_still_works(self, tmp_path):
        from generative.calibration.sample import candidate_notes

        folder = tmp_path / "Legacy-Stem"
        folder.mkdir()
        (folder / "vault__Legacy.md").write_text("legacy", encoding="utf-8")

        assert candidate_notes(folder) == [folder / "vault__Legacy.md"]


class TestReevalBaselineMixedLegacyFlat:
    """#261: Im GEMISCHTEN Ordner (Legacy-flat `vault__*.md` direkt unter <stem>/
    PLUS >=1 run_id-Unterordner) wurden die Legacy-flat-Notes auf master still
    verworfen, sobald ein run_id-Ordner existierte — der gesamte pre-#241-
    Baseline-Bestand fiel damit beim ersten Neu-Lauf aus reeval/Kalibrierung
    (realer Datenverlust: der Produktions-Cache ist aktuell komplett Legacy-flat).
    Fix (b): Legacy-flat-Notes zusaetzlich einbeziehen, dedupliziert per Dateiname
    (die Note des neuesten run_id-Ordners hat Vorrang)."""

    def test_mixed_folder_includes_both_legacy_flat_and_run_id_notes(self, tmp_path):
        from generative.reeval_baseline import _baseline_note_files

        pdf_dir = tmp_path / "Mixed-Stem"
        pdf_dir.mkdir()
        (pdf_dir / "vault__Legacy.md").write_text("legacy", encoding="utf-8")  # pre-#241 flat
        run_dir = pdf_dir / "20260113-120000"
        run_dir.mkdir()
        (run_dir / "vault__New.md").write_text("new", encoding="utf-8")

        files = _baseline_note_files(pdf_dir)

        assert sorted(p.name for p in files) == ["vault__Legacy.md", "vault__New.md"]
        assert (pdf_dir / "vault__Legacy.md") in files
        assert (run_dir / "vault__New.md") in files

    def test_mixed_folder_dedupes_same_filename_run_id_wins(self, tmp_path):
        from generative.reeval_baseline import _baseline_note_files

        pdf_dir = tmp_path / "Mixed-Dup-Stem"
        pdf_dir.mkdir()
        (pdf_dir / "vault__Dup.md").write_text("legacy-body", encoding="utf-8")
        run_dir = pdf_dir / "20260113-120000"
        run_dir.mkdir()
        (run_dir / "vault__Dup.md").write_text("run-body", encoding="utf-8")

        files = _baseline_note_files(pdf_dir)

        assert files == [run_dir / "vault__Dup.md"]
        assert files[0].read_text(encoding="utf-8") == "run-body"

    def test_pure_legacy_and_pure_nested_unchanged(self, tmp_path):
        from generative.reeval_baseline import _baseline_note_files

        legacy = tmp_path / "Legacy-Only"
        legacy.mkdir()
        (legacy / "vault__L.md").write_text("l", encoding="utf-8")
        assert _baseline_note_files(legacy) == [legacy / "vault__L.md"]

        nested = tmp_path / "Nested-Only"
        nested.mkdir()
        run_dir = nested / "20260113-120000"
        run_dir.mkdir()
        (run_dir / "vault__N.md").write_text("n", encoding="utf-8")
        assert _baseline_note_files(nested) == [run_dir / "vault__N.md"]


class TestCalibrationSampleMixedLegacyFlat:
    """#261 fuer calibration/sample.candidate_notes(): dieselbe Regel — Legacy-flat
    zusaetzlich einbeziehen, per Dateiname dedupliziert (run_id gewinnt)."""

    def test_mixed_folder_includes_both_legacy_flat_and_run_id_notes(self, tmp_path):
        from generative.calibration.sample import candidate_notes

        folder = tmp_path / "Mixed-Stem"
        folder.mkdir()
        (folder / "vault__Legacy.md").write_text("legacy", encoding="utf-8")
        (folder / "inbox__LegacyInbox.md").write_text("legacy2", encoding="utf-8")
        run_dir = folder / "20260113-120000"
        run_dir.mkdir()
        (run_dir / "vault__New.md").write_text("new", encoding="utf-8")

        notes = candidate_notes(folder)

        assert sorted(p.name for p in notes) == [
            "inbox__LegacyInbox.md",
            "vault__Legacy.md",
            "vault__New.md",
        ]

    def test_mixed_folder_dedupes_same_filename_run_id_wins(self, tmp_path):
        from generative.calibration.sample import candidate_notes

        folder = tmp_path / "Mixed-Dup-Stem"
        folder.mkdir()
        (folder / "vault__Dup.md").write_text("legacy-body", encoding="utf-8")
        run_dir = folder / "20260113-120000"
        run_dir.mkdir()
        (run_dir / "vault__Dup.md").write_text("run-body", encoding="utf-8")

        notes = candidate_notes(folder)

        assert notes == [run_dir / "vault__Dup.md"]
        assert notes[0].read_text(encoding="utf-8") == "run-body"
