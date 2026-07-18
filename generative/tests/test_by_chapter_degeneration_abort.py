"""#345: --by-chapter bricht bei einem degenerierten Split (keine nutzbare Outline)
HART ab, statt mit Hunderten Mikro-„Kapiteln" loszulaufen (der 15,5-h-/32-Mio-Token-/
0-Notes-Lauf). Ein echter Outline-Split (source="outline") passiert den Guard."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from generative import orchestrator
from generative.schemas.atomic_note import ConceptPlan, QualityReport


def _stub(monkeypatch, chunks):
    pc = orchestrator.pdf_chunker
    monkeypatch.setattr(pc, "pdf_to_text", lambda *_a, **_k: "Quelltext mit genug Woertern fuer den Test.")
    monkeypatch.setattr(pc, "split_by_chapters", lambda *_a, **_k: chunks)
    monkeypatch.setattr(pc, "extract_overview", lambda *_a, **_k: "Überblick")
    monkeypatch.setattr(
        pc, "pdf_metadata", lambda *_a, **_k: {"Author": "A", "Year": "2020", "Title": "T", "Pages": "500"}
    )
    monkeypatch.setattr(orchestrator.acronym_fix, "extract_acronym_pairs", lambda *_a, **_k: {})
    monkeypatch.setattr(
        orchestrator.context_builder,
        "build_relevance_profile",
        lambda *_a, **_k: {"existing_concepts": [], "tag_whitelist": []},
    )
    monkeypatch.setattr(orchestrator.context_builder, "build_concept_links", lambda *_a, **_k: {})
    monkeypatch.setattr(
        orchestrator.quality,
        "check_quality",
        lambda **_kw: QualityReport(peer_reviewed=None, citation_count=None, retracted=False, flags=[]),
    )
    monkeypatch.setattr(orchestrator.planner, "run", lambda *_a, **_k: ConceptPlan("T", "S", []))
    monkeypatch.setattr(orchestrator.planner, "filter_hallucinated", lambda plan, _text: (plan, []))

    async def _no_concepts(*_a, **_k):
        return ([], {}, 0, [])

    monkeypatch.setattr(orchestrator, "run_extractors_per_concept", _no_concepts)


def _chunks(n, source):
    return [SimpleNamespace(title=f"Seg {i}", text=f"Text {i} " * 20, source=source) for i in range(n)]


def test_by_chapter_aborts_on_degenerate_non_outline_split(monkeypatch):
    """Viele Chunks OHNE Outline (source="words") unter --by-chapter → SystemExit."""
    _stub(monkeypatch, _chunks(120, "words"))
    args = SimpleNamespace(by_chapter=True, dry_run=True, doi=None, llm_fallback=False)
    with pytest.raises(SystemExit) as exc:
        orchestrator._run_extraction_stages(args, Path("fake.pdf"), None)
    assert "degenerierter Split" in str(exc.value)


def test_by_chapter_proceeds_on_outline_split(monkeypatch):
    """Viele Chunks MIT Outline (source="outline") → kein Abbruch (echter Buch-Split)."""
    _stub(monkeypatch, _chunks(30, "outline"))
    args = SimpleNamespace(by_chapter=True, dry_run=True, doi=None, llm_fallback=False)
    # Läuft durch (kein SystemExit); Rückgabe ist die RunContext-Dataclass.
    result = orchestrator._run_extraction_stages(args, Path("fake.pdf"), None)
    assert result is not None


def test_normal_path_unaffected_by_guard(monkeypatch):
    """Ohne --by-chapter bleibt selbst ein großer Nicht-Outline-Split erlaubt (Warnung, kein Abbruch)."""
    _stub(monkeypatch, _chunks(120, "words"))
    args = SimpleNamespace(by_chapter=False, dry_run=True, doi=None, llm_fallback=False)
    result = orchestrator._run_extraction_stages(args, Path("fake.pdf"), None)
    assert result is not None
