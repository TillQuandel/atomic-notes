"""Tests für #283: n_merge muss den tatsächlichen Merge-AUSGANG zählen
(Merge-Stub-Erzeugung + Sibling-Dedup-Absorption), nicht die Planner-VORAB-
Klassifikation (`action == "extend"`).

Belegter Bug (Testlauf-Serie 2026-07-14, Issue #283): Lauf 2 hatte 1 echten
Merge-Stub im Log, DB `n_merge=0` (Draft war als "create" geplant, wurde aber
erst beim Schreiben zum Merge-Stub). Lauf 5 hatte 2 echte Sibling-Dedup-Merges
bei 0 geplanten "extend"-Konzepten, DB `n_merge=0`. Lauf 4s `n_merge=2` war
zufällig korrekt (geplante extends == spätere Merge-Ziele).

`count_actual_merges(drafts, n_sibling_dedup)` ersetzt die alte Intent-Zählung:
- Sibling-Dedup-Absorption kommt als Parameter rein (die absorbierten Drafts
  sind zum Zeitpunkt des Aufrufs schon aus `drafts` entfernt — resolve_sibling_dups
  gibt ihre Anzahl bereits zurück, siehe test_sibling_dedup.py).
- Merge-Stub-Erzeugung wird auf den POST-Write-Drafts über `is_merge_stub`
  ausgewertet (vault_writer.write_note setzt dieses Feld beim Schreiben,
  siehe test_typeaware_dedup.py).
"""

from __future__ import annotations

from generative.orchestrator import count_actual_merges
from generative.schemas.atomic_note import AtomicNoteDraft


def _draft(title, action="create", is_merge_stub=False) -> AtomicNoteDraft:
    d = AtomicNoteDraft(
        title=title,
        body="Body.",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high",
        action=action,
    )
    d.is_merge_stub = is_merge_stub
    return d


def test_no_merges_is_zero():
    drafts = [_draft("A"), _draft("B")]
    assert count_actual_merges(drafts, n_sibling_dedup=0) == 0


def test_counts_write_time_merge_stub():
    # Draft war als "create" geplant, wurde aber beim Schreiben zum Merge-Stub
    # (Title-/Alias-Match im Vault entdeckt) — der alte Bug hätte das NICHT gezählt.
    drafts = [_draft("A", action="create", is_merge_stub=True), _draft("B", action="create")]
    assert count_actual_merges(drafts, n_sibling_dedup=0) == 1


def test_counts_sibling_dedup_absorption_via_parameter():
    # 2 echte Sibling-Dedup-Merges, 0 geplante extends (Lauf-5-Repro) — die
    # absorbierten Drafts sind hier schon aus `drafts` raus, n_sibling_dedup trägt sie.
    drafts = [_draft("Survivor", action="create")]
    assert count_actual_merges(drafts, n_sibling_dedup=2) == 2


def test_planned_extend_without_actual_merge_not_counted():
    # Planner-Intent "extend", aber resolve_sibling_dups hat es auf "create"
    # zurückgesetzt (dangling Intra-Run-extend) UND write_note fand keinen
    # Vault-Treffer -> kein echter Merge. Alte Zählung (action=="extend") hätte
    # hier faelschlich 1 gezaehlt.
    drafts = [_draft("A", action="extend", is_merge_stub=False)]
    assert count_actual_merges(drafts, n_sibling_dedup=0) == 0


def test_sums_both_sources_without_double_counting():
    drafts = [
        _draft("A", action="create", is_merge_stub=True),
        _draft("B", action="extend", is_merge_stub=True),
        _draft("C", action="extend", is_merge_stub=False),
        _draft("D", action="create", is_merge_stub=False),
    ]
    assert count_actual_merges(drafts, n_sibling_dedup=3) == 2 + 3
