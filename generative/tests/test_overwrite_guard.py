"""Tests für den content-hash-Überschreib-Schutz (#47).

Ein Re-Run desselben PDFs darf eine vom Nutzer editierte Inbox-Note nicht still
überschreiben (Datenverlust). content-hash-Guard: pristine Pipeline-Notes dürfen
idempotent überschrieben werden, editierte nicht.
"""

from generative.schemas.atomic_note import AtomicNoteDraft


def _draft(title="Test Note", body="pipeline body"):
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


class TestContentHashHelpers:
    def test_injected_note_is_pristine(self):
        from generative.pipeline.vault_writer import inject_content_hash, is_pristine_pipeline_note

        text = inject_content_hash("---\ntitle: x\n---\nbody\n")
        assert "pipeline-content-hash:" in text
        assert is_pristine_pipeline_note(text) is True

    def test_edited_note_is_not_pristine(self):
        from generative.pipeline.vault_writer import inject_content_hash, is_pristine_pipeline_note

        text = inject_content_hash("---\ntitle: x\n---\nbody\n")
        edited = text.replace("body", "body\nUSER-EDIT")
        assert is_pristine_pipeline_note(edited) is False

    def test_note_without_hash_is_not_pristine(self):
        # konservativ: alte Notes ohne Hash gelten als editiert → schützen
        from generative.pipeline.vault_writer import is_pristine_pipeline_note

        assert is_pristine_pipeline_note("---\ntitle: x\n---\nbody\n") is False


class TestRenderNoteHasHash:
    def test_render_note_embeds_content_hash(self):
        from generative.pipeline.vault_writer import render_note, is_pristine_pipeline_note

        out = render_note(_draft(), "src.pdf")
        assert "pipeline-content-hash:" in out
        assert is_pristine_pipeline_note(out) is True


class TestHashScopedToFrontmatter:
    def test_body_line_with_hash_field_does_not_fool_check(self):
        # ein im Body editierter Text, der zufällig wie das Hash-Feld aussieht,
        # darf den Pristine-Check nicht aushebeln (Strip nur im Frontmatter).
        from generative.pipeline.vault_writer import inject_content_hash, is_pristine_pipeline_note

        text = inject_content_hash("---\ntitle: x\n---\nbody\n")
        edited = text.replace("body\n", "body\npipeline-content-hash: fake\n")
        assert is_pristine_pipeline_note(edited) is False


class TestHubAndStubGetHash:
    def test_render_moc_embeds_hash(self):
        from generative.pipeline.vault_writer import render_moc, is_pristine_pipeline_note

        hub = AtomicNoteDraft(
            title="MoC Test",
            body="hub body",
            source_anchors=[],
            related=[],
            tags=["t"],
            synthesis_confidence="low",
            action="hub",
            hub_subconcepts=["A", "B"],
            critic_score=5,
            hard_gates_pass=True,
        )
        out = render_moc(hub, "src.pdf")
        assert "pipeline-content-hash:" in out
        assert is_pristine_pipeline_note(out) is True


class TestMergeStubOverwriteGuard:
    def test_edited_merge_stub_not_overwritten(self, tmp_path, monkeypatch):
        from generative.pipeline import vault_writer

        # VAULT auf tmp umbiegen + bestehende Vault-Note anlegen (löst merge-stub aus)
        monkeypatch.setattr(vault_writer, "VAULT", tmp_path)
        vault_note = tmp_path / "Existing.md"
        vault_note.write_text('---\ntitle: "Test Note"\nsource-file: "other.pdf"\n---\nx\n', encoding="utf-8")
        monkeypatch.setattr(vault_writer, "find_existing_in_vault", lambda *a, **k: vault_note)
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        # 1. erster Merge-Stub-Write
        p1 = vault_writer.write_note(
            _draft(), source_file="src.pdf", dry_run=False, existing_concepts={"x": "y"}, inbox_dir=inbox
        )
        assert "MERGE" in p1.name
        # 2. Nutzer editiert den Stub
        p1.write_text(p1.read_text(encoding="utf-8") + "\nUSER-EDIT\n", encoding="utf-8")
        # 3. Re-Run
        p2 = vault_writer.write_note(
            _draft(), source_file="src.pdf", dry_run=False, existing_concepts={"x": "y"}, inbox_dir=inbox
        )
        assert "USER-EDIT" in p1.read_text(encoding="utf-8")
        assert p2 != p1


class TestReRunOverwriteGuard:
    def test_edited_inbox_note_not_overwritten(self, tmp_path):
        from generative.pipeline import vault_writer

        # 1. erster Pipeline-Write
        p1 = vault_writer.write_note(_draft(body="ORIGINAL"), source_file="src.pdf", dry_run=False, inbox_dir=tmp_path)
        # 2. Nutzer editiert den Body
        txt = p1.read_text(encoding="utf-8")
        p1.write_text(txt.replace("ORIGINAL", "ORIGINAL\nUSER-HANDEDITIERT"), encoding="utf-8")
        # 3. Re-Run derselben PDF
        p2 = vault_writer.write_note(
            _draft(body="NEUER RERUN BODY"), source_file="src.pdf", dry_run=False, inbox_dir=tmp_path
        )
        # User-Edit muss überleben, neue Version landet in separater Datei
        assert "USER-HANDEDITIERT" in p1.read_text(encoding="utf-8")
        assert p2 != p1
        assert "NEUER RERUN BODY" in p2.read_text(encoding="utf-8")

    def test_finder_prefers_pristine_match_regardless_of_order(self, tmp_path):
        # Bei mehreren Treffern (editiertes Original + pristine Variante) muss der
        # Finder die pristine Datei wählen — deterministisch, unabhängig von der
        # glob-Reihenfolge (sonst Variant-Churn auf manchen Dateisystemen).
        from generative.pipeline import vault_writer

        edited = tmp_path / "aaa.md"  # sortiert zuerst
        edited.write_text('---\ntitle: "T"\nsource-file: "s.pdf"\n---\nedited\n', encoding="utf-8")
        pristine = tmp_path / "zzz.md"  # sortiert zuletzt
        pristine.write_text(
            vault_writer.inject_content_hash('---\ntitle: "T"\nsource-file: "s.pdf"\n---\nclean\n'), encoding="utf-8"
        )
        assert vault_writer.find_existing_in_inbox("s.pdf", "T", tmp_path) == pristine

    def test_repeated_reruns_reuse_pristine_variant_no_churn(self, tmp_path):
        # Wiederholte Re-Runs über eine editierte Note dürfen nicht endlos neue
        # Varianten (-3, -4, …) erzeugen — die pristine Variante wird idempotent
        # überschrieben (kein Churn).
        from generative.pipeline import vault_writer

        p1 = vault_writer.write_note(_draft(body="ORIGINAL"), source_file="src.pdf", dry_run=False, inbox_dir=tmp_path)
        # Nutzer editiert das Original
        p1.write_text(p1.read_text(encoding="utf-8").replace("ORIGINAL", "ORIGINAL\nUSER"), encoding="utf-8")
        p2 = vault_writer.write_note(
            _draft(body="RERUN A"), source_file="src.pdf", dry_run=False, inbox_dir=tmp_path
        )  # → Variante
        p3 = vault_writer.write_note(
            _draft(body="RERUN B"), source_file="src.pdf", dry_run=False, inbox_dir=tmp_path
        )  # → pristine Variante überschreiben
        assert p3 == p2
        assert "RERUN B" in p3.read_text(encoding="utf-8")
        assert "USER" in p1.read_text(encoding="utf-8")  # Original-Edit weiter sicher
        assert len(list(tmp_path.glob("*.md"))) == 2  # nur Original + eine Variante

    def test_pristine_inbox_note_still_overwritten(self, tmp_path):
        # Idempotenz bleibt: unveränderte Pipeline-Note wird überschrieben
        from generative.pipeline import vault_writer

        p1 = vault_writer.write_note(_draft(body="ORIGINAL"), source_file="src.pdf", dry_run=False, inbox_dir=tmp_path)
        p2 = vault_writer.write_note(
            _draft(body="NEUER RERUN BODY"), source_file="src.pdf", dry_run=False, inbox_dir=tmp_path
        )
        assert p2 == p1  # gleiche Datei überschrieben
        assert "NEUER RERUN BODY" in p2.read_text(encoding="utf-8")
        assert len(list(tmp_path.glob("*.md"))) == 1
