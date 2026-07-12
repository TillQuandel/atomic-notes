"""Issue #210: CLI-Timeout darf Konzepte nicht still verschlucken.

Deckt die zwei neuen Verhaltensweisen ab:
1. run_extractors_per_concept trennt echte Call-Fehler (Timeout/Exception) von
   legitimen Leer-Extraktionen (run_per_concept liefert None) und meldet erstere
   als `failures` zurück.
2. Reine Helfer, die aus den Failures den Warn-Block fürs Run-Summary und den
   Exit-Code (3 = Konzepte verloren, unterscheidbar von hartem Fehler=1) bauen.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import generative.agents.extractor as extractor_mod
from generative.orchestrator import (
    run_extractors_per_concept,
    format_extractor_failure_report,
    extractor_failure_exit_code,
)
from generative.schemas.atomic_note import AtomicNoteDraft, ConceptItem, ConceptPlan


def _draft(title: str) -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title=title,
        body=f"Body zu {title} mit ausreichend Text.",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high",
    )


def _plan(*titles: str) -> ConceptPlan:
    return ConceptPlan(
        source_title="Testquelle",
        source_summary="Zwei Konzepte.",
        concepts=[ConceptItem(title=t, priority="high", chapter="1", action="create") for t in titles],
    )


_FULL_TEXT = (
    "Alpha Concept ist ein zentrales Thema in diesem Dokument und wird hier ausführlich "
    "behandelt. Danach folgt Beta Concept, das ebenfalls im Volltext erscheint und mehrfach "
    "diskutiert wird. Alpha Concept und Beta Concept stehen im Zentrum der Analyse."
)


def test_run_extractors_reports_call_failures_separately():
    """Timeout in Call 1, Erfolg in Call 2 → 1 Draft, dropped=1, failures nennt Konzept 1."""
    plan = _plan("Alpha Concept", "Beta Concept")

    async def fake_run_per_concept(*, concept, **kwargs):
        if concept.title == "Alpha Concept":
            raise RuntimeError("claude CLI Timeout nach 300s (extractor/anthropic/claude-sonnet-4-6)")
        return _draft(concept.title)

    with patch.object(extractor_mod, "run_per_concept", side_effect=fake_run_per_concept):
        drafts, concept_map, dropped, failures = asyncio.run(
            run_extractors_per_concept(_FULL_TEXT, plan, existing_concepts={})
        )

    assert [d.title for d in drafts] == ["Beta Concept"]
    assert dropped == 1
    assert len(failures) == 1
    failed_title, failed_err = failures[0]
    assert failed_title == "Alpha Concept"
    assert "Timeout" in failed_err


def test_empty_extraction_is_not_counted_as_failure():
    """run_per_concept=None (Konzept zu schwach im Text) ist KEIN Fehler → failures leer."""
    plan = _plan("Alpha Concept", "Beta Concept")

    async def fake_run_per_concept(*, concept, **kwargs):
        if concept.title == "Alpha Concept":
            return None  # legitime Leer-Extraktion
        return _draft(concept.title)

    with patch.object(extractor_mod, "run_per_concept", side_effect=fake_run_per_concept):
        drafts, concept_map, dropped, failures = asyncio.run(
            run_extractors_per_concept(_FULL_TEXT, plan, existing_concepts={})
        )

    assert [d.title for d in drafts] == ["Beta Concept"]
    assert dropped == 1  # Leer-Extraktion zählt weiter als dropped (n_dropped)
    assert failures == []  # aber NICHT als Fehler


def test_all_calls_fail_yields_no_drafts_and_two_failures():
    plan = _plan("Alpha Concept", "Beta Concept")

    async def fake_run_per_concept(*, concept, **kwargs):
        raise RuntimeError("claude CLI Timeout nach 300s (extractor/x)")

    with patch.object(extractor_mod, "run_per_concept", side_effect=fake_run_per_concept):
        drafts, concept_map, dropped, failures = asyncio.run(
            run_extractors_per_concept(_FULL_TEXT, plan, existing_concepts={})
        )

    assert drafts == []
    assert dropped == 2
    assert {t for t, _ in failures} == {"Alpha Concept", "Beta Concept"}


# --- reine Helfer: Warn-Block + Exit-Code ---


def test_failure_report_empty_when_no_failures():
    assert format_extractor_failure_report([], n_attempted=3) == []


def test_failure_report_names_count_denominator_and_timeout_reason():
    lines = format_extractor_failure_report(
        [("Alpha Concept", "claude CLI Timeout nach 300s (extractor/x)")],
        n_attempted=2,
    )
    assert lines  # nicht leer
    head = lines[0]
    assert "1" in head and "2" in head  # "1 von 2"
    body = "\n".join(lines)
    assert "Alpha Concept" in body
    assert "Timeout" in body


def test_failure_report_marks_non_timeout_as_generic_error():
    lines = format_extractor_failure_report(
        [("Beta Concept", "claude CLI fehlgeschlagen (rc=1): boom")],
        n_attempted=1,
    )
    body = "\n".join(lines)
    assert "Beta Concept" in body
    assert "Fehler" in body


def test_exit_code_is_zero_without_failures():
    assert extractor_failure_exit_code([]) == 0


def test_exit_code_is_three_when_concepts_lost():
    assert extractor_failure_exit_code([("Alpha Concept", "Timeout")]) == 3
