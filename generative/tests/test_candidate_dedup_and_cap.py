"""Tests für die isolierten Hybrid-Buchplanungs-Bausteine (#346, PR 2/3).

Zwei reine Funktionen in orchestrator.py, die PR 3 später verdrahtet:

- ``dedup_concept_candidates`` — globale Kandidaten-Dedup über normalisierten
  Titel; stärkere Fassung (Priorität, dann non-skip) gewinnt; secondary_mention
  ist ein eigener Kanal (nie dedupliziert); je verworfenem Duplikat ein
  ``planner_dedup``-Funnel-Event.
- ``cap_candidates_balanced`` — wortanteil-balancierter, prioritätssortierter
  Kandidaten-Cap; highs zuerst über alle Kapitel, Überlauf proportional zum
  Wortanteil gekürzt; Rest-Slots nach Wortanteil verteilt; secondary_mention
  budget-neutral.

Backend-/Capture-Muster wie test_stage_outcome_events.py: echtes JsonlBackend
auf tmp umbiegen, JSONL zurücklesen.
"""

from __future__ import annotations

import json

import generative.agents.tracing as tracing
from generative.agents.tracing import JsonlBackend
from generative import orchestrator as orch
from generative.schemas.atomic_note import ConceptItem


# --- Helfer -----------------------------------------------------------------


def _concept(
    title: str,
    priority: str = "medium",
    chapter: str = "K1",
    action: str = "create",
    origin: str = "primary",
) -> ConceptItem:
    return ConceptItem(
        title=title,
        priority=priority,
        chapter=chapter,
        action=action,
        origin=origin,
    )


def _capture(monkeypatch, tmp_path):
    """Biegt das Trace-Backend auf tmp um; gibt einen Reader für stage_outcome-Events zurück."""
    backend = JsonlBackend(run_dir=tmp_path, run_id="test-run")
    monkeypatch.setattr(tracing, "_backend", backend)

    def _read_stage_events() -> list[dict]:
        f = tmp_path / "test-run.jsonl"
        if not f.exists():
            return []
        events = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [e for e in events if e.get("type") == "stage_outcome"]

    return _read_stage_events


# ===========================================================================
# dedup_concept_candidates
# ===========================================================================


def test_dedup_single_occurrence_no_drop():
    c = _concept("Data Governance")
    kept, dropped = orch.dedup_concept_candidates([c])
    assert dropped == 0
    assert [x.title for x in kept] == ["Data Governance"]


def test_dedup_normalized_duplicate_across_chapters():
    # Gleicher normalisierter Titel (Case + Satzzeichen), verschiedene Kapitel,
    # gleiche Priorität/Action -> Erstauftreten überlebt.
    a = _concept("Data Governance", chapter="Kap1")
    b = _concept("data governance!", chapter="Kap3")
    kept, dropped = orch.dedup_concept_candidates([a, b])
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0].title == "Data Governance"  # Erstauftreten
    assert kept[0].chapter == "Kap1"


def test_dedup_priority_upgrade_replaces_survivor():
    a = _concept("X", priority="medium")
    b = _concept("X", priority="high")
    kept, dropped = orch.dedup_concept_candidates([a, b])
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0].priority == "high"  # stärkere Fassung gewinnt


def test_dedup_skip_to_actionable_upgrade():
    a = _concept("Y", priority="medium", action="skip")
    b = _concept("Y", priority="medium", action="create")
    kept, dropped = orch.dedup_concept_candidates([a, b])
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0].action == "create"  # non-skip schlägt skip bei gleicher Prio


def test_dedup_actionable_survivor_not_replaced_by_skip():
    # Umkehrung: Erstauftreten actionable, zweites skip bei gleicher Prio -> bleibt.
    a = _concept("Y", priority="medium", action="create")
    b = _concept("Y", priority="medium", action="skip")
    kept, dropped = orch.dedup_concept_candidates([a, b])
    assert dropped == 1
    assert kept[0].action == "create"


def test_dedup_secondary_mention_untouched():
    # Zwei secondary_mention mit gleichem Titel -> beide bleiben (eigener Kanal).
    a = _concept("S", origin="secondary_mention")
    b = _concept("S", origin="secondary_mention")
    # zusätzlich ein primary mit gleichem normalisierten Titel -> kollidiert NICHT
    p = _concept("S", origin="primary")
    kept, dropped = orch.dedup_concept_candidates([a, p, b])
    assert dropped == 0
    assert len(kept) == 3


def test_dedup_funnel_event_emitted(monkeypatch, tmp_path):
    read = _capture(monkeypatch, tmp_path)
    survivor = _concept("Metadaten", priority="high", chapter="Kap1")
    dup = _concept("metadaten", priority="low", chapter="Kap2")  # schwächer -> verworfen

    kept, dropped = orch.dedup_concept_candidates([survivor, dup])

    assert dropped == 1
    assert kept[0].priority == "high"
    events = read()
    assert len(events) == 1
    e = events[0]
    assert e["title"] == "metadaten"  # der verworfene Kandidat
    assert e["stage"] == "planner_dedup"
    assert e["outcome"] == "dropped"
    assert e["drop_reason"] == "chapter_duplicate"
    assert e["detail"] == "survivor=Metadaten"


def test_dedup_empty_input():
    kept, dropped = orch.dedup_concept_candidates([])
    assert kept == []
    assert dropped == 0


# ===========================================================================
# cap_candidates_balanced
# ===========================================================================


def test_cap_highs_overflow_proportional_to_word_share():
    # 3 highs Kapitel A (700 W) + 3 highs Kapitel B (300 W), Budget 4.
    # D'Hondt-Verteilung -> A=3, B=1.
    concepts = [
        _concept("A1", priority="high", chapter="A"),
        _concept("A2", priority="high", chapter="A"),
        _concept("A3", priority="high", chapter="A"),
        _concept("B1", priority="high", chapter="B"),
        _concept("B2", priority="high", chapter="B"),
        _concept("B3", priority="high", chapter="B"),
    ]
    keys = ["A", "A", "A", "B", "B", "B"]
    wc = {"A": 700, "B": 300}
    kept, capped = orch.cap_candidates_balanced(concepts, 4, wc, keys)

    assert len(kept) == 4
    assert len(capped) == 2
    a_kept = sum(1 for c in kept if c.chapter == "A")
    b_kept = sum(1 for c in kept if c.chapter == "B")
    assert a_kept == 3  # großes Kapitel bekommt mehr Slots
    assert b_kept == 1
    assert all(c.chapter == "B" for c in capped)


def test_cap_rest_slots_by_word_share():
    # Nur medium-Kandidaten -> alle Slots über die Rest-Verteilung.
    # A (800 W) 6 medium, B (200 W) 6 medium, Budget 5 -> A=4, B=1.
    concepts = [_concept(f"A{i}", priority="medium", chapter="A") for i in range(6)]
    concepts += [_concept(f"B{i}", priority="medium", chapter="B") for i in range(6)]
    keys = ["A"] * 6 + ["B"] * 6
    wc = {"A": 800, "B": 200}
    kept, capped = orch.cap_candidates_balanced(concepts, 5, wc, keys)

    assert len(kept) == 5
    a_kept = sum(1 for c in kept if c.chapter == "A")
    b_kept = sum(1 for c in kept if c.chapter == "B")
    assert a_kept > b_kept  # großes Kapitel mehr Slots
    assert a_kept == 4
    assert b_kept == 1


def test_cap_priority_within_chapter():
    # Ein Kapitel, Budget 2, Mix aus low/high/medium -> high + medium bleiben.
    low = _concept("low", priority="low", chapter="K1")
    high = _concept("high", priority="high", chapter="K1")
    medium = _concept("medium", priority="medium", chapter="K1")
    concepts = [low, high, medium]
    keys = ["K1", "K1", "K1"]
    wc = {"K1": 500}
    kept, capped = orch.cap_candidates_balanced(concepts, 2, wc, keys)

    kept_titles = {c.title for c in kept}
    assert kept_titles == {"high", "medium"}
    assert [c.title for c in capped] == ["low"]


def test_cap_secondary_mention_budget_neutral():
    concepts = [
        _concept("A1", priority="high", chapter="A"),
        _concept("A2", priority="high", chapter="A"),
        _concept("S", origin="secondary_mention", chapter="A"),
    ]
    keys = ["A", "A", "A"]
    wc = {"A": 100}
    kept, capped = orch.cap_candidates_balanced(concepts, 1, wc, keys)

    kept_titles = {c.title for c in kept}
    # secondary_mention immer behalten, budget-neutral; nur 1 high passt ins Budget
    assert "S" in kept_titles
    high_kept = sum(1 for c in kept if c.origin == "primary")
    assert high_kept == 1
    assert len(capped) == 1
    assert all(c.origin != "secondary_mention" for c in capped)


def test_cap_budget_exceeds_candidate_count_is_noop():
    concepts = [
        _concept("A1", priority="high", chapter="A"),
        _concept("B1", priority="medium", chapter="B"),
    ]
    keys = ["A", "B"]
    wc = {"A": 100, "B": 100}
    kept, capped = orch.cap_candidates_balanced(concepts, 100, wc, keys)

    assert capped == []
    assert [id(c) for c in kept] == [id(c) for c in concepts]  # unverändert


def test_cap_deterministic_on_tie():
    # Gleiche Wortzahl + gleiche Priorität -> deterministisch (frühere Reihenfolge).
    concepts = [
        _concept("A1", priority="medium", chapter="A"),
        _concept("B1", priority="medium", chapter="B"),
        _concept("A2", priority="medium", chapter="A"),
        _concept("B2", priority="medium", chapter="B"),
    ]
    keys = ["A", "B", "A", "B"]
    wc = {"A": 500, "B": 500}
    kept1, capped1 = orch.cap_candidates_balanced(concepts, 2, wc, keys)
    kept2, capped2 = orch.cap_candidates_balanced(concepts, 2, wc, keys)

    assert [c.title for c in kept1] == [c.title for c in kept2]  # stabil
    assert len(kept1) == 2
    # Tie-Break: je ein Slot pro Kapitel, Erstauftreten
    assert {c.chapter for c in kept1} == {"A", "B"}
