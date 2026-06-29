"""Tests für den Overwrite-Diff vor dem Vault-Write (#46).

Schlanker Markdown-Diff gegen die bestehende Note im Overwrite-Fall — begrenzt
auf den Overwrite-Pfad, kein Voll-Diff-UI.
"""


def _draft(title="Test Note", body="neuer body"):
    from generative.schemas.atomic_note import AtomicNoteDraft

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


class TestDryRunOverwriteWiring:
    def test_dry_run_prints_diff_when_overwriting_existing_inbox(self, tmp_path, capsys):
        from generative.pipeline import vault_writer

        # Bestehende, PRISTINE Pipeline-Note (mit gültigem content-hash, sonst greift
        # der #47-Überschreib-Schutz und es käme zu keinem Overwrite/Diff).
        existing = tmp_path / "test-note.md"
        existing.write_text(
            vault_writer.inject_content_hash('---\ntitle: "Test Note"\nsource-file: "src.pdf"\n---\nalter body\n'),
            encoding="utf-8",
        )
        vault_writer.write_note(
            _draft(body="komplett neuer body"), source_file="src.pdf", dry_run=True, inbox_dir=tmp_path
        )
        out = capsys.readouterr().out
        assert "Overwrite-Diff" in out
        assert "-" in out and "+" in out  # alte + neue Zeilen

    def test_dry_run_no_diff_for_fresh_note(self, tmp_path, capsys):
        from generative.pipeline import vault_writer

        vault_writer.write_note(_draft(), source_file="fresh.pdf", dry_run=True, inbox_dir=tmp_path)
        out = capsys.readouterr().out
        assert "Overwrite-Diff" not in out


class TestMarkdownOverwriteDiff:
    def test_identical_content_returns_empty(self):
        from generative.pipeline.vault_writer import markdown_overwrite_diff

        assert markdown_overwrite_diff("a\nb\n", "a\nb\n") == ""

    def test_changed_line_shows_old_and_new(self):
        from generative.pipeline.vault_writer import markdown_overwrite_diff

        out = markdown_overwrite_diff("title: alt\nbody\n", "title: neu\nbody\n")
        assert out != ""
        assert "-title: alt" in out
        assert "+title: neu" in out

    def test_added_line_marked(self):
        from generative.pipeline.vault_writer import markdown_overwrite_diff

        out = markdown_overwrite_diff("a\n", "a\nb\n")
        assert "+b" in out

    def test_slim_caps_huge_diffs(self):
        # schlank: ein riesiger Diff wird gekappt, kein Voll-Diff-UI
        from generative.pipeline.vault_writer import markdown_overwrite_diff

        old = "\n".join(f"old{i}" for i in range(500))
        new = "\n".join(f"new{i}" for i in range(500))
        out = markdown_overwrite_diff(old, new)
        assert out.count("\n") < 80
        assert "…" in out or "..." in out  # Kürzungs-Hinweis
