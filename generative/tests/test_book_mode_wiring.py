"""#346: --book-mode — Planung je Hauptkapitel, Extraktion global.

Kern-Kontrakt (drei Text-Scopes, explizit getestet):
- Normalpfad extrahiert aus dem Volltext (plant auf Overview).
- --by-chapter extrahiert je Kapitel aus chunk.text (partitioniert auch die Extraktion).
- --book-mode plant lokal je Hauptkapitel, extrahiert EINMAL global über den Volltext
  (Notes können über Kapitelgrenzen synthetisieren — das ist der Kern-Kontrakt).

Muster: test_by_chapter_background_extractor_skip.py (Stub der Stage-1–5-Senken,
Integration über _run_extraction_stages).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from generative import orchestrator
from generative.schemas.atomic_note import ConceptItem, ConceptPlan, QualityReport


def _chunk(title: str, text: str, source: str) -> SimpleNamespace:
    return SimpleNamespace(title=title, text=text, source=source)


def _concept(title: str, *, priority: str = "high", action: str = "create", origin: str = "primary") -> ConceptItem:
    return ConceptItem(title=title, priority=priority, chapter="", action=action, origin=origin)


def _install(monkeypatch, chunks, plans, *, gate: bool = False):
    """Stubt die Stage-1–5-Senken und gibt (calls, full_text) zurück.

    plans: dict plan_text -> Konzeptliste (planner.run liefert daraus). full_text ist
    der von pdf_to_text zurückgegebene Volltext (Kern-Assertion des Extractor-Scopes).
    """
    pc = orchestrator.pdf_chunker
    full_text = "VOLLTEXT_WORT " * 80
    monkeypatch.setattr(pc, "pdf_to_text", lambda *_a, **_k: full_text)
    monkeypatch.setattr(pc, "split_by_chapters", lambda *_a, **_k: chunks)
    monkeypatch.setattr(
        pc, "pdf_metadata", lambda *_a, **_k: {"Author": "Autor", "Year": "2020", "Title": "Titel", "Pages": "300"}
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

    calls: dict = {"planner": [], "extract": []}

    def _planner_run(plan_text, _profile, **_kw):
        calls["planner"].append(plan_text)
        return ConceptPlan("Titel", "Summary", list(plans.get(plan_text, [])))

    monkeypatch.setattr(orchestrator.planner, "run", _planner_run)
    monkeypatch.setattr(orchestrator.planner, "filter_hallucinated", lambda plan, _text: (plan, []))

    async def _extract(full, plan, *_a, **_k):
        calls["extract"].append(
            SimpleNamespace(
                full_text=full,
                concepts=list(plan.concepts),
                related=list(_k.get("related_mentions") or []),
            )
        )
        return ([], {}, 0, [])  # #210: 4. Rückgabewert = extractor_failures

    monkeypatch.setattr(orchestrator, "run_extractors_per_concept", _extract)
    monkeypatch.setattr(orchestrator, "ENABLE_BACKGROUND_EXTRACTOR", gate)
    return calls, full_text


def _args(**over) -> SimpleNamespace:
    base = dict(book_mode=False, by_chapter=False, dry_run=True, doi=None, llm_fallback=False)
    base.update(over)
    return SimpleNamespace(**base)


# --- Kern-Kontrakt: drei Text-Scopes -------------------------------------------


def test_normal_path_extracts_from_full_text(monkeypatch):
    chunks = [_chunk("Abschnitt 1", "abschnitt eins " * 40, "words")]
    calls, full_text = _install(monkeypatch, chunks, {"Überblick": [_concept("Konzept X")]})

    orchestrator._run_extraction_stages(_args(), Path("paper.pdf"), None)

    assert len(calls["planner"]) == 1
    assert calls["planner"][0] == "Überblick"  # Normalpfad plant auf Overview
    assert len(calls["extract"]) == 1
    assert calls["extract"][0].full_text == full_text  # Extractor über den Volltext


def test_by_chapter_extracts_from_chunk_text(monkeypatch):
    chunks = [
        _chunk("Kapitel 1", "KAPITEL_EINS " * 40, "outline"),
        _chunk("Kapitel 2", "KAPITEL_ZWEI " * 40, "outline"),
    ]
    plans = {chunks[0].text: [_concept("A")], chunks[1].text: [_concept("B")]}
    calls, _ = _install(monkeypatch, chunks, plans)

    orchestrator._run_extraction_stages(_args(by_chapter=True), Path("book.pdf"), None)

    # Ein Planner- UND ein Extractor-Call je Kapitel; extrahiert aus chunk.text (nicht Volltext).
    assert len(calls["planner"]) == 2
    assert len(calls["extract"]) == 2
    assert {c.full_text for c in calls["extract"]} == {chunks[0].text, chunks[1].text}


def test_book_mode_plans_locally_extracts_globally(monkeypatch):
    """Kern-Regressionstest: book-mode plant je Kapitel, extrahiert EINMAL mit Volltext."""
    chunks = [
        _chunk("Kapitel 1", "kap eins " * 40, "outline"),
        _chunk("Kapitel 2", "kap zwei " * 40, "outline"),
    ]
    plans = {
        chunks[0].text: [_concept("Konzept A"), _concept("Konzept B")],
        chunks[1].text: [_concept("Konzept C")],
    }
    calls, full_text = _install(monkeypatch, chunks, plans)

    orchestrator._run_extraction_stages(_args(book_mode=True), Path("book.pdf"), None)

    assert len(calls["planner"]) == 2  # Planner je Hauptkapitel
    assert len(calls["extract"]) == 1  # EIN globaler Extractor-Lauf
    assert calls["extract"][0].full_text == full_text  # … mit dem Volltext
    titles = {c.title for c in calls["extract"][0].concepts}
    assert titles == {"Konzept A", "Konzept B", "Konzept C"}  # Kandidaten aus beiden Kapiteln


# --- Fallback / Wiring ---------------------------------------------------------


def test_book_mode_without_outline_falls_back_to_normal(monkeypatch, capsys):
    chunks = [
        _chunk("Abschnitt 1", "a " * 40, "heuristic"),
        _chunk("Abschnitt 2", "b " * 40, "heuristic"),
    ]
    calls, full_text = _install(monkeypatch, chunks, {"Überblick": [_concept("Konzept X")]})

    orchestrator._run_extraction_stages(_args(book_mode=True), Path("nooutline.pdf"), None)

    # Kein Outline-Split → transparenter Normalpfad: Planner genau 1x auf Overview.
    assert len(calls["planner"]) == 1
    assert calls["planner"][0] == "Überblick"
    assert len(calls["extract"]) == 1
    assert calls["extract"][0].full_text == full_text
    assert "book-mode" in capsys.readouterr().out.lower()  # sichtbare Diagnose


def test_planner_calls_equal_chapters_and_candidates_within_budget(monkeypatch):
    monkeypatch.setattr(orchestrator, "BOOK_MODE_CONCEPTS_PER_CHAPTER", 2)
    monkeypatch.setattr(orchestrator, "BOOK_MODE_MAX_TOTAL", 100)
    chunks = [
        _chunk("Kapitel 1", "eins " * 60, "outline"),
        _chunk("Kapitel 2", "zwei " * 20, "outline"),
    ]
    plans = {
        chunks[0].text: [_concept(f"K1-{i}") for i in range(3)],
        chunks[1].text: [_concept(f"K2-{i}") for i in range(3)],
    }
    calls, _ = _install(monkeypatch, chunks, plans)

    orchestrator._run_extraction_stages(_args(book_mode=True), Path("book.pdf"), None)

    assert len(calls["planner"]) == 2  # Planner-Calls == Kapitelzahl
    budget = min(2 * 2, 100)  # = 4
    assert len(calls["extract"][0].concepts) == budget  # Kandidaten ≤ Budget (hier == 4)


def test_book_mode_filters_skip_actions_before_cap(monkeypatch):
    """Skip-action-Kandidaten VOR dem Cap heraus — sie erreichen den Extractor nie."""
    chunks = [
        _chunk("Kapitel 1", "eins " * 40, "outline"),
        _chunk("Kapitel 2", "zwei " * 40, "outline"),
    ]
    plans = {
        chunks[0].text: [_concept("Keep A"), _concept("Skip Me", action="skip")],
        chunks[1].text: [_concept("Keep B")],
    }
    calls, _ = _install(monkeypatch, chunks, plans)

    orchestrator._run_extraction_stages(_args(book_mode=True), Path("book.pdf"), None)

    titles = {c.title for c in calls["extract"][0].concepts}
    assert titles == {"Keep A", "Keep B"}
    assert "Skip Me" not in titles


def test_book_mode_dedups_secondary_mentions_across_chapters(monkeypatch):
    chunks = [
        _chunk("Kapitel 1", "eins " * 40, "outline"),
        _chunk("Kapitel 2", "zwei " * 40, "outline"),
    ]
    plans = {
        chunks[0].text: [_concept("Primär A"), _concept("Wilson", origin="secondary_mention")],
        chunks[1].text: [_concept("Primär B"), _concept("Wilson", origin="secondary_mention")],
    }
    calls, _ = _install(monkeypatch, chunks, plans)

    orchestrator._run_extraction_stages(_args(book_mode=True), Path("book.pdf"), None)

    # Sekundär-Erwähnung nicht als primäres Konzept extrahiert …
    assert {c.title for c in calls["extract"][0].concepts} == {"Primär A", "Primär B"}
    # … sondern EINMAL (kapitelübergreifend dedupliziert) in related_mentions.
    assert calls["extract"][0].related.count("Wilson") == 1


def test_book_mode_skips_background_extractor(monkeypatch):
    """Background-Extractor läuft im book-mode nie — auch bei aktivem Gate (#102-Analogie)."""
    chunks = [
        _chunk("Kapitel 1", "eins " * 40, "outline"),
        _chunk("Kapitel 2", "zwei " * 40, "outline"),
    ]
    plans = {chunks[0].text: [_concept("A")], chunks[1].text: [_concept("B")]}
    calls, _ = _install(monkeypatch, chunks, plans, gate=True)

    def _boom(*_a, **_k):
        raise AssertionError("Background-Extractor darf im book-mode nie laufen")

    monkeypatch.setattr(orchestrator.background_extractor, "run", _boom)

    orchestrator._run_extraction_stages(_args(book_mode=True), Path("book.pdf"), None)

    assert len(calls["extract"]) == 1  # kein Crash → Background-Extractor nie aufgerufen


def test_book_mode_and_by_chapter_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        orchestrator.main(["--source", "x.pdf", "--by-chapter", "--book-mode"])


# --- PR-2-Nachzügler (Mistral-Review): cap_candidates_balanced-Randfälle --------


def test_cap_budget_zero_keeps_only_secondary_mentions():
    concepts = [
        _concept("A", origin="primary"),
        _concept("B", origin="primary"),
        _concept("S", origin="secondary_mention"),
    ]
    keys = ["Kap1", "Kap1", "Kap1"]

    kept, capped = orchestrator.cap_candidates_balanced(concepts, 0, {"Kap1": 100}, keys)

    assert [c.title for c in kept] == ["S"]  # nur secondary_mention überlebt
    assert {c.title for c in capped} == {"A", "B"}


def test_cap_missing_chapter_word_count_no_crash():
    """Kapitel-Key ohne chapter_word_counts-Eintrag → get(k, 0)-Fallback, kein KeyError."""
    concepts = [_concept("A"), _concept("B"), _concept("C")]
    keys = ["Kap1", "Kap2", "Kap2"]  # keiner in chapter_word_counts

    kept, capped = orchestrator.cap_candidates_balanced(concepts, 2, {}, keys)

    assert len(kept) == 2  # deterministisch gekappt statt Crash
    assert len(kept) + len(capped) == 3
