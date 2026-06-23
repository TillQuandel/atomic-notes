"""Tests für typ-bewusstes Dedup-Blocking (Konzept-Note ≠ Duplikat ihrer Lit-Note).

Ebner-Run 2026-06-23: die Konzept-Note „Webinar" wurde als Duplikat-Risiko-hoch gegen die
existierende `type: literature`-Note ba-lit-ebner-gegenfurtner-2019 geflaggt — obwohl Lit-
und Konzept-Notes per Vault-Design (Schema-Lit vs. Schema-Konzept) koexistieren. Ursache:
der Dedup-Pool/-Lookup war typ-blind. Fix: literature/moc/merge-stub sind keine Dup-/Merge-
Kandidaten; related-Links über Typgrenzen bleiben erlaubt.
"""
from generative.agents.context_builder import is_dedup_eligible, note_type, resolve_vault_relpath
from generative.schemas.atomic_note import AtomicNoteDraft


def _write_note(path, type_, title="X"):
    # Titel quoten — reale Notes tun das (Umlaute/Doppelpunkte im Titel), sonst bricht YAML.
    path.write_text(f'---\ntype: {type_}\ntitle: "{title}"\n---\nKörper-Text.', encoding="utf-8")


def _draft(title, action="create", extend_path=None):
    return AtomicNoteDraft(
        title=title, body="Körper-Text der Note.", source_anchors=[], related=[],
        tags=[], synthesis_confidence="high", aliases=[],
        action=action, extend_path=extend_path)


def test_note_type_reads_frontmatter(tmp_path):
    p = tmp_path / "n.md"
    _write_note(p, "Literature")  # Groß/Klein egal
    assert note_type(p) == "literature"


def test_is_dedup_eligible_excludes_literature_moc_stub(tmp_path):
    for t, expected in [("literature", False), ("moc", False), ("merge-stub", False),
                        ("atomic", True), ("note", True), ("", True)]:
        p = tmp_path / f"{t or 'empty'}.md"
        _write_note(p, t)
        assert is_dedup_eligible(p) is expected, t


def test_find_existing_in_vault_skips_literature(tmp_path, monkeypatch):
    import generative.pipeline.vault_writer as vw
    monkeypatch.setattr(vw, "VAULT", tmp_path)
    _write_note(tmp_path / "ba-lit-ebner.md", "literature", title="BA — Lit: Ebner")
    ec = {"webinar": "ba-lit-ebner.md"}  # Titel/Alias-Treffer zeigt auf eine Lit-Note
    assert vw.find_existing_in_vault("Webinar", [], ec) is None


def test_find_existing_in_vault_returns_concept(tmp_path, monkeypatch):
    import generative.pipeline.vault_writer as vw
    monkeypatch.setattr(vw, "VAULT", tmp_path)
    _write_note(tmp_path / "webinar.md", "atomic", title="Webinar")
    ec = {"webinar": "webinar.md"}
    assert vw.find_existing_in_vault("Webinar", [], ec) == tmp_path / "webinar.md"


def test_find_existing_alias_finds_concept_after_skipping_lit(tmp_path, monkeypatch):
    # Titel trifft eine Lit-Note (übersprungen), ein Alias trifft eine echte Konzept-Note.
    import generative.pipeline.vault_writer as vw
    monkeypatch.setattr(vw, "VAULT", tmp_path)
    _write_note(tmp_path / "lit.md", "literature")
    _write_note(tmp_path / "concept.md", "atomic")
    ec = {"webinar": "lit.md", "synchrones online-lehrformat": "concept.md"}
    got = vw.find_existing_in_vault("Webinar", ["synchrones Online-Lehrformat"], ec)
    assert got == tmp_path / "concept.md"


def test_cross_reference_dup_target_eligible(tmp_path, monkeypatch):
    from generative.agents import cross_reference as cr
    monkeypatch.setattr(cr, "VAULT", tmp_path)
    _write_note(tmp_path / "ba-lit-ebner-gegenfurtner-2019.md", "literature")
    _write_note(tmp_path / "affective-access.md", "atomic")
    ec = {
        "ba — lit: ebner & gegenfurtner (2019)": "ba-lit-ebner-gegenfurtner-2019.md",
        "affective access": "affective-access.md",
    }
    # dup_path wie vom LLM geliefert: der Datei-Stem (so werden Kandidaten präsentiert).
    assert cr._dup_target_eligible("ba-lit-ebner-gegenfurtner-2019", ec) is False  # Lit → kein Dup
    assert cr._dup_target_eligible("affective-access", ec) is True                 # Konzept → Dup ok
    assert cr._dup_target_eligible("Nicht-Im-Vault-Sibling", ec) is True           # unauflösbar → unverändert
    assert cr._dup_target_eligible("affective access", ec) is True                 # Titel-Treffer (nicht Stem)


# --- #2b: write_note honoriert extend_path typ-sicher (Vault-Konzept-Dup mit anderem Titel) ---

def test_resolve_vault_relpath_stem_title_wikilink():
    ec = {"wilson information need": "04-wissen/wilson-information-need.md"}
    assert resolve_vault_relpath("wilson-information-need", ec) == "04-wissen/wilson-information-need.md"  # Stem
    assert resolve_vault_relpath("Wilson Information Need", ec) == "04-wissen/wilson-information-need.md"  # Titel
    assert resolve_vault_relpath("[[wilson-information-need]]", ec) == "04-wissen/wilson-information-need.md"  # Wikilink
    assert resolve_vault_relpath("Unbekannt", ec) is None
    assert resolve_vault_relpath("", ec) is None


def test_resolve_vault_relpath_ambiguous_stem_returns_none():
    # Zwei Notes mit gleichem Datei-Stem in verschiedenen Ordnern → mehrdeutig → None
    # (kein willkürlicher Treffer → kein Fehl-Merge). Titel-Treffer bleibt eindeutig.
    ec = {"konzept a": "04-wissen/webinar.md", "konzept b": "01-studium/webinar.md"}
    assert resolve_vault_relpath("webinar", ec) is None
    assert resolve_vault_relpath("Konzept A", ec) == "04-wissen/webinar.md"


def test_write_note_honors_extend_path_to_concept(tmp_path, monkeypatch):
    # Draft-Titel ≠ Vault-Titel, aber extend_path zeigt (per Stem) auf eine Konzept-Note.
    # find_existing_in_vault matcht nicht → früher Dublette; jetzt via extend_path → Merge-Stub.
    import generative.pipeline.vault_writer as vw
    monkeypatch.setattr(vw, "VAULT", tmp_path)
    _write_note(tmp_path / "wilson-information-need.md", "atomic", title="Wilson Information Need")
    inbox = tmp_path / "inbox"; inbox.mkdir()
    ec = {"wilson information need": "wilson-information-need.md"}
    d = _draft("Information Need", action="extend", extend_path="wilson-information-need")
    target = vw.write_note(d, source_file="X.pdf", dry_run=False, existing_concepts=ec, inbox_dir=inbox)
    assert "MERGE" in target.name      # Merge-Stub statt create-Dublette
    assert target.exists()


def test_write_note_extend_path_to_literature_not_merged(tmp_path, monkeypatch):
    # Defense-in-depth: zeigt extend_path (entgegen #2a) doch auf eine Lit-Note → KEIN Merge.
    import generative.pipeline.vault_writer as vw
    monkeypatch.setattr(vw, "VAULT", tmp_path)
    _write_note(tmp_path / "ba-lit-x.md", "literature", title="BA Lit X")
    inbox = tmp_path / "inbox"; inbox.mkdir()
    ec = {"ba lit x": "ba-lit-x.md"}
    d = _draft("Webinar", action="extend", extend_path="ba-lit-x")
    target = vw.write_note(d, source_file="X.pdf", dry_run=False, existing_concepts=ec, inbox_dir=inbox)
    assert "MERGE" not in target.name  # keine Lit-Note als Merge-Ziel → normale create-Note
    assert target.exists()
