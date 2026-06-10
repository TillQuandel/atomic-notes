"""Tests für Schwäche-4-Fix: Tag-Whitelist mit Source-Scoring + Bootstrap-Pfad.

Konservative Variante (Codex-Empfehlung + Literatur-Konvergenz):
- Whitelist bleibt closed-set (Anti-Halluzination, Auto-Note-Mover-Routing)
- Source-Scoring sortiert die Whitelist im Prompt in zwei Blöcke (Bias-Fix)
- proposed-tags als getrenntes Schema-Feld für Bootstrap, KEIN Routing
"""
from __future__ import annotations
import sys
from pathlib import Path


from generative.agents.context_builder import score_tags_for_source
from generative.agents.extractor import _validate_proposed_tags, _format_tag_whitelist


# ---- Source-Scoring (Schwäche 4a: Bias-Fix) ---------------------------------

def test_score_finds_lexical_match():
    whitelist = ["change-management", "information-behavior", "methods"]
    prio, rest = score_tags_for_source(whitelist, "ADKAR is a change management framework.")
    assert "change-management" in prio
    assert "information-behavior" in rest


def test_score_handles_hierarchical_tag():
    whitelist = ["change-management/adkar", "uni/ibi/konzept"]
    prio, rest = score_tags_for_source(whitelist, "Discussion of ADKAR change phases.")
    assert "change-management/adkar" in prio
    assert "uni/ibi/konzept" in rest


def test_score_empty_source_returns_all_as_rest():
    whitelist = ["change-management", "methods"]
    prio, rest = score_tags_for_source(whitelist, "")
    assert prio == []
    assert set(rest) == set(whitelist)


def test_score_no_match_returns_all_as_rest():
    whitelist = ["bibliometrics", "ontology"]
    prio, rest = score_tags_for_source(whitelist, "Cooking recipes and cake baking instructions.")
    assert prio == []
    assert set(rest) == set(whitelist)


def test_score_stopwords_ignored():
    # 'der die das und mit' sind Stoppwörter, sollen keinen Tag triggern können
    whitelist = ["der-und", "framework"]
    prio, rest = score_tags_for_source(whitelist, "Der und die und das mit dem.")
    # `der-und`-Tokens = {"der", "und"} — beide Stoppwörter, kein Source-Match
    # → rest, nicht prio
    assert "der-und" in rest
    assert "framework" in rest
    assert prio == []


def test_score_priority_by_overlap_count():
    # Mehr Token-Overlap → höherer Rang
    whitelist = ["change", "change-management", "change-management-process"]
    prio, _ = score_tags_for_source(
        whitelist, "Change management process for organizational change."
    )
    # change-management-process matched alle drei Tokens, sollte vor "change" stehen
    assert prio[0] == "change-management-process"


# ---- Format-Output für Prompt -----------------------------------------------

def test_format_whitelist_two_blocks_when_source_given():
    whitelist = ["change-management", "uni/ibi/konzept", "methods"]
    out = _format_tag_whitelist(whitelist, source_text="ADKAR change management awareness phase.")
    assert "Quellnah" in out
    assert "Übrige" in out
    # Reihenfolge im Output: priorisierter Block vor übrigem
    assert out.index("Quellnah") < out.index("Übrige")


def test_format_whitelist_flat_when_no_source():
    whitelist = ["change-management", "methods"]
    out = _format_tag_whitelist(whitelist, source_text=None)
    assert "Quellnah" not in out
    assert "- change-management" in out
    assert "- methods" in out


def test_format_whitelist_empty():
    assert "Tag-Feld leer lassen" in _format_tag_whitelist([], source_text="anything")
    assert "Tag-Feld leer lassen" in _format_tag_whitelist(None)


# ---- Proposed-Tag-Validator (Schwäche 4b: Bootstrap) ------------------------

def test_validator_accepts_kebab_case_hierarchical():
    out = _validate_proposed_tags(["change-management/adkar"], whitelist=set())
    assert out == ["change-management/adkar"]


def test_validator_rejects_uppercase():
    out = _validate_proposed_tags(["ChangeManagement"], whitelist=set())
    assert out == []


def test_validator_rejects_whitespace():
    out = _validate_proposed_tags(["change management"], whitelist=set())
    assert out == []


def test_validator_rejects_too_deep_hierarchy():
    out = _validate_proposed_tags(["a/b/c/d"], whitelist=set())
    assert out == []


def test_validator_drops_tags_already_in_whitelist():
    # Wenn Tag schon Whitelist-Mitglied ist, gehört er nach `tags:`, nicht
    # nach `proposed-tags:`. Bootstrap-Feld nur für wirklich NEUE Tags.
    out = _validate_proposed_tags(["change-management"], whitelist={"change-management"})
    assert out == []


def test_validator_max_count_two():
    out = _validate_proposed_tags(
        ["one", "two", "three", "four"], whitelist=set(), max_count=2
    )
    assert len(out) == 2


def test_validator_silent_on_bad_input():
    # LLM-Tippfehler dürfen Pipeline nicht crashen
    assert _validate_proposed_tags([None, 42, "valid-tag"], whitelist=set()) == ["valid-tag"]
    assert _validate_proposed_tags("not a list", whitelist=set()) == []


def test_validator_strips_hash_prefix():
    out = _validate_proposed_tags(["#change-management"], whitelist=set())
    assert out == ["change-management"]


# ---- Codex-Schwäche4-Review Folge-Fixes -------------------------------------

def test_validator_rejects_trailing_hyphen():
    # Codex-Finding 2: `bad-` durfte unter alter Regex durch
    assert _validate_proposed_tags(["bad-"], whitelist=set()) == []


def test_validator_rejects_double_hyphen():
    assert _validate_proposed_tags(["a--b"], whitelist=set()) == []


def test_validator_rejects_empty_hierarchy_segment():
    assert _validate_proposed_tags(["a//b"], whitelist=set()) == []


def test_validator_rejects_pure_numbers():
    # Segment muss mit Buchstabe starten
    assert _validate_proposed_tags(["123"], whitelist=set()) == []
    assert _validate_proposed_tags(["a/123"], whitelist=set()) == []


def test_validator_accepts_numbers_in_segment():
    # Aber Ziffern innerhalb Segments sind ok
    assert _validate_proposed_tags(["adkar-step1"], whitelist=set()) == ["adkar-step1"]


def test_render_moc_includes_proposed_tags():
    """Codex-Finding 5: render_moc rendert proposed-tags-Block."""
    from generative.pipeline.vault_writer import render_moc
    from generative.schemas.atomic_note import AtomicNoteDraft
    note = AtomicNoteDraft(
        title="ADKAR Model", body="# Hub\n\nEinleitung.", source_anchors=[],
        related=[], tags=[], synthesis_confidence="medium",
        action="hub", hub_subconcepts=["Awareness", "Desire"],
        proposed_tags=["change-management/adkar"],
        tag_review_status="needs-review",
    )
    out = render_moc(note, source_file="hiatt-2006.pdf")
    assert "proposed-tags:" in out
    assert "- change-management/adkar" in out
    assert "tag-review-status: needs-review" in out


def test_render_note_skips_proposed_block_when_empty():
    from generative.pipeline.vault_writer import render_note
    from generative.schemas.atomic_note import AtomicNoteDraft
    note = AtomicNoteDraft(
        title="Test", body="# Test\n\nBody.", source_anchors=[],
        related=[], tags=["methods"], synthesis_confidence="medium",
    )
    out = render_note(note, source_file="test.pdf")
    assert "proposed-tags:" not in out
    assert "tag-review-status" not in out


def test_header_normalizes_dash_to_underscore():
    """Zusatzbefund: Modell schreibt `proposed-tags:`, Parser muss das aufnehmen."""
    from generative.agents.structured_output import parse_extractor_output
    text = (
        "<!--NOTE-->\n"
        "title: Test\n"
        "tags: methods\n"
        "proposed-tags: change-management/adkar\n"
        "<!--BODY-->\n"
        "Body.\n"
        "<!--END-->\n"
    )
    notes, _ = parse_extractor_output(text)
    assert notes[0]["proposed_tags"] == ["change-management/adkar"]


def test_registry_loader_handles_top_level_list():
    """Codex-Finding 7: malformed YAML (Top-Level-Liste statt Dict) → kein Crash."""
    from generative.agents import context_builder
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write("- just\n- a\n- list\n")
        path = Path(f.name)
    orig = context_builder.TAG_REGISTRY_PATH
    try:
        context_builder.TAG_REGISTRY_PATH = path
        assert context_builder._load_registry_tags() == set()
    finally:
        context_builder.TAG_REGISTRY_PATH = orig
        path.unlink()


def test_registry_loader_validates_schema():
    """Registry-Einträge durchlaufen denselben Schema-Validator wie proposed-tags."""
    from generative.agents import context_builder
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write("approved:\n  - valid-tag\n  - bad-\n  - a//b\n  - UPPER\n  - ok/deep/tag\n")
        path = Path(f.name)
    orig = context_builder.TAG_REGISTRY_PATH
    try:
        context_builder.TAG_REGISTRY_PATH = path
        tags = context_builder._load_registry_tags()
        assert "valid-tag" in tags
        assert "ok/deep/tag" in tags
        assert "bad-" not in tags
        assert "a//b" not in tags
        assert "UPPER" not in tags
    finally:
        context_builder.TAG_REGISTRY_PATH = orig
        path.unlink()


def test_source_scoring_filters_generic_tokens():
    """Codex-Finding 3: generische Tokens wie `model`/`method`/`theory` dürfen
    nicht alleine einen Tag priorisieren."""
    whitelist = ["change-management", "model-theory", "ontology"]
    # Source enthält nur generische Tokens — kein spezifisches Match möglich
    prio, rest = score_tags_for_source(whitelist, "This research uses a theory and a model.")
    assert "model-theory" not in prio, "Generic-only Tags dürfen nicht priorisiert werden"
    assert "model-theory" in rest


def test_source_scoring_keeps_specific_when_generic_present():
    """Wenn ein Tag GENERIC+SPECIFIC kombiniert (z.B. `change-management`), zählt
    nur das spezifische Token zur Priorisierung."""
    whitelist = ["change-management", "ontology"]
    prio, _ = score_tags_for_source(whitelist, "Discussion of change processes in organizations.")
    assert "change-management" in prio, "specific token 'change' muss matchen, auch wenn 'management' generic ist"


def test_source_scoring_mojibake_robust():
    """Replacement-Zeichen aus Mojibake-Source werden gefiltert, Tokenisierung läuft trotzdem."""
    whitelist = ["change-management"]
    # Mojibake-Source: enthält U+FFFD-Zeichen statt Umlauten
    prio, _ = score_tags_for_source(whitelist, "Change�management�process�im�Unternehmen.")
    assert "change-management" in prio


def test_inbox_reread_preserves_proposed_tags_block():
    """Codex-Finding 1: Re-Run ohne neue Vorschläge bewahrt User-Review-State."""
    from generative.pipeline.vault_writer import _read_proposed_tags_from_inbox
    import tempfile
    from pathlib import Path
    content = """---
title: "Test"
tags:
  - methods
proposed-tags:
  - change-management/adkar
  - data-engineering
tag-review-status: needs-review
---
# Test
Body.
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = Path(f.name)
    try:
        tags, status = _read_proposed_tags_from_inbox(path)
        assert tags == ["change-management/adkar", "data-engineering"]
        assert status == "needs-review"
    finally:
        path.unlink()


def test_inbox_reread_handles_missing_file():
    from generative.pipeline.vault_writer import _read_proposed_tags_from_inbox
    from pathlib import Path
    tags, status = _read_proposed_tags_from_inbox(Path("/nonexistent/file.md"))
    assert tags == []
    assert status is None


def test_inbox_reread_handles_no_proposed_block():
    from generative.pipeline.vault_writer import _read_proposed_tags_from_inbox
    import tempfile
    from pathlib import Path
    content = """---
title: "Test"
tags:
  - methods
---
Body without proposed-tags.
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = Path(f.name)
    try:
        tags, status = _read_proposed_tags_from_inbox(path)
        assert tags == []
        assert status is None
    finally:
        path.unlink()


def test_registry_loader_missing_file_returns_empty():
    from generative.agents import context_builder
    from pathlib import Path
    orig = context_builder.TAG_REGISTRY_PATH
    try:
        context_builder.TAG_REGISTRY_PATH = Path("/nonexistent/path/registry.yml")
        assert context_builder._load_registry_tags() == set()
    finally:
        context_builder.TAG_REGISTRY_PATH = orig
