"""#102: --by-chapter deaktiviert den Background-Extractor (Stage 4.5) still.

Maintainer-Entscheidung: Variante (a) — Sichtbarkeit statt Verdrahtung.
Background-Calls würden sich pro Kapitel multiplizieren (Kosten); das ist eine
bewusste Scope-Entscheidung, keine Wiring-Lücke. Fix: Log-Zeile + Kommentar,
analog zum Single-Doc-else-Zweig ("[4.5/7] ... deaktiviert (...)").

Variante (b) — echte Verdrahtung pro Kapitel — ist bewusst NICHT Teil dieses
Fixes (eigenes Feature-Issue bei Bedarf).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from generative import orchestrator
from generative.schemas.atomic_note import ConceptPlan, QualityReport


# --- Pure Helper ---


def test_skip_line_present_when_gate_enabled():
    msg = orchestrator._background_extractor_by_chapter_skip_line(gate_enabled=True)
    assert msg is not None
    assert "Background-Extractor" in msg
    assert "by-chapter" in msg


def test_skip_line_absent_when_gate_disabled():
    """Kein Skip-Hinweis wenn das Gate ohnehin aus ist — sonst doppelt-verwirrend
    (Background-Extractor liefe so oder so nicht)."""
    assert orchestrator._background_extractor_by_chapter_skip_line(gate_enabled=False) is None


# --- Integrationstest über _run_extraction_stages (by-chapter-Pfad) ---


def _stub_common(monkeypatch, n_chapters: int):
    pc = orchestrator.pdf_chunker
    monkeypatch.setattr(pc, "pdf_to_text", lambda *_a, **_k: "Etwas Quelltext mit genug Woertern fuer den Test.")
    monkeypatch.setattr(
        pc,
        "split_by_chapters",
        lambda *_a, **_k: [SimpleNamespace(title=f"Kapitel {i}", text=f"Kapiteltext {i}") for i in range(n_chapters)],
    )
    monkeypatch.setattr(
        pc, "pdf_metadata", lambda *_a, **_k: {"Author": "Autor", "Year": "2020", "Title": "Titel", "Pages": "1"}
    )
    monkeypatch.setattr(pc, "extract_overview", lambda *_a, **_k: "Überblick")
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
    monkeypatch.setattr(orchestrator.planner, "run", lambda *_a, **_k: ConceptPlan("Titel", "Summary", []))
    monkeypatch.setattr(orchestrator.planner, "filter_hallucinated", lambda plan, _text: (plan, []))

    async def _no_concepts(*_a, **_k):
        return ([], {}, 0)

    monkeypatch.setattr(orchestrator, "run_extractors_per_concept", _no_concepts)


def test_by_chapter_with_gate_on_prints_skip_line(monkeypatch, capsys):
    _stub_common(monkeypatch, n_chapters=2)
    monkeypatch.setattr(orchestrator, "ENABLE_BACKGROUND_EXTRACTOR", True)

    args = SimpleNamespace(by_chapter=True, dry_run=True, doi=None, llm_fallback=False)
    orchestrator._run_extraction_stages(args, Path("fake.pdf"), None)

    out = capsys.readouterr().out
    assert "Background-Extractor" in out
    assert "by-chapter" in out
    # Nur EINE Skip-Zeile für den gesamten Lauf — nicht pro Kapitel gespammt.
    assert out.count("Background-Extractor: übersprungen") == 1


def test_by_chapter_with_gate_off_behaves_as_before(monkeypatch, capsys):
    _stub_common(monkeypatch, n_chapters=2)
    monkeypatch.setattr(orchestrator, "ENABLE_BACKGROUND_EXTRACTOR", False)

    args = SimpleNamespace(by_chapter=True, dry_run=True, doi=None, llm_fallback=False)
    orchestrator._run_extraction_stages(args, Path("fake.pdf"), None)

    out = capsys.readouterr().out
    assert "übersprungen" not in out
