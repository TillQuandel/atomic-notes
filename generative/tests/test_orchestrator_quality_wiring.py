"""Regressionstest: das G6-Textqualitäts-Gate darf das `quality`-Modul nicht shadowen.

Der G6-Commit (#27) führte in `_run_extraction_stages` die lokale Bindung
`quality = pdf_chunker.assess_text_quality(text)` ein und überschrieb damit das
oben importierte Modul `generative.agents.quality`. Der spätere Aufruf
`quality.check_quality(...)` traf dann die TextQuality-Instanz statt das Modul
→ `AttributeError: 'TextQuality' object has no attribute 'check_quality'`,
jeder Lauf starb in Stage 3.

Dieser Test fährt `_run_extraction_stages` mit gestubbten I/O-/LLM-Kollaborateuren
bis zum Quality-Agent-Aufruf und prüft, dass das MODUL `quality.check_quality`
erreicht wird (Wiring-Test — pure-Helper-Tests fingen den Bug nicht).
"""
from pathlib import Path
from types import SimpleNamespace

from generative import orchestrator
from generative.schemas.atomic_note import ConceptPlan, QualityReport


def test_quality_module_not_shadowed_by_text_quality_gate(monkeypatch):
    pc = orchestrator.pdf_chunker
    monkeypatch.setattr(pc, "pdf_to_text", lambda *_a, **_k: "Etwas Quelltext mit Wörtern.")
    # assess_text_quality bewusst ECHT lassen — reproduziert die TextQuality-Bindung,
    # die den Bug ausgelöst hat.
    monkeypatch.setattr(pc, "split_by_chapters",
                        lambda *_a, **_k: [SimpleNamespace(title="Intro", text="body")])
    monkeypatch.setattr(pc, "pdf_metadata",
                        lambda *_a, **_k: {"Author": "Autor", "Year": "2020",
                                          "Title": "Titel", "Pages": "1"})
    monkeypatch.setattr(pc, "extract_overview", lambda *_a, **_k: "Überblick")
    monkeypatch.setattr(orchestrator.acronym_fix, "extract_acronym_pairs", lambda *_a, **_k: {})
    monkeypatch.setattr(orchestrator.context_builder, "build_relevance_profile",
                        lambda *_a, **_k: {"existing_concepts": [], "tag_whitelist": []})
    monkeypatch.setattr(orchestrator.context_builder, "build_concept_links",
                        lambda *_a, **_k: {})
    monkeypatch.setattr(orchestrator, "ENABLE_BACKGROUND_EXTRACTOR", False)
    monkeypatch.setattr(orchestrator.planner, "run",
                        lambda *_a, **_k: ConceptPlan("Titel", "Summary", []))
    monkeypatch.setattr(orchestrator.planner, "filter_hallucinated",
                        lambda plan, _text: (plan, []))

    async def _no_concepts(*_a, **_k):
        return ([], {}, 0)
    monkeypatch.setattr(orchestrator, "run_extractors_per_concept", _no_concepts)

    calls = {"n": 0}

    def _spy_check_quality(**_kw):
        calls["n"] += 1
        return QualityReport(peer_reviewed=None, citation_count=None,
                             retracted=False, flags=[])
    monkeypatch.setattr(orchestrator.quality, "check_quality", _spy_check_quality)

    args = SimpleNamespace(by_chapter=False, dry_run=True, doi=None, llm_fallback=False)

    result = orchestrator._run_extraction_stages(args, Path("fake.pdf"), None)

    # Der Quality-Agent (Modul) muss genau einmal erreicht worden sein.
    assert calls["n"] == 1
    # quality_report wandert an Tupel-Position 7.
    assert isinstance(result[7], QualityReport)
