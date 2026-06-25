from __future__ import annotations

from pathlib import Path


from generative.orchestrator import dry_run_eval_targets


def test_excludes_stale_notes_from_previous_runs(tmp_path):
    # Notes dieses Laufs im Cache-Ordner
    (tmp_path / "vault__A.md").write_text("a", encoding="utf-8")
    (tmp_path / "inbox__B.md").write_text("b", encoding="utf-8")
    # Note aus einem fruheren Lauf — liegt im selben Ordner, darf NICHT mit-evaluiert werden
    (tmp_path / "vault__OLD.md").write_text("old", encoding="utf-8")

    written = [
        (Path("A.md"), True),    # vault-empfohlen -> evaluieren
        (Path("B.md"), False),   # Inbox -> nicht evaluieren
    ]

    assert dry_run_eval_targets(written, tmp_path) == [tmp_path / "vault__A.md"]


def test_skips_missing_eval_files(tmp_path):
    # vault-empfohlen, aber keine vault__-Datei vorhanden (z.B. Merge-Stub mit "merge"-Prefix)
    written = [(Path("Ghost.md"), True)]

    assert dry_run_eval_targets(written, tmp_path) == []
