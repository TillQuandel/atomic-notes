"""Tests fürs Faithfulness-Gate-Wiring in Pipeline/Routing (E6, #69).

Das Gate selbst (`run_faithfulness_gate`) ist fertig getestet in
`test_faithfulness_gate.py` — hier geht es nur um den Kleber:

- Routing-Veto in `auto_write_decision` (vault_writer.py)
- Gate-Aufruf + Draft-Mutation in `_apply_faithfulness_gate` (orchestrator.py)

Nutzt die echten Dataclasses (`GateResult`/`ClaimVerdict`/`Claim`), keine
Fake-Klassen — Feld-Drift zwischen Gate und Wiring würde sonst nicht auffallen.
"""

from __future__ import annotations

import generative.orchestrator as orchestrator
from generative.pipeline.claims import Claim
from generative.pipeline.faithfulness_gate import ClaimVerdict, GateResult
from generative.pipeline.vault_writer import auto_write_decision
from generative.schemas.atomic_note import AtomicNoteDraft
from generative.schemas.citation import CitationMeta


def _draft(**kw) -> AtomicNoteDraft:
    base = dict(
        title="T",
        body="b",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="low",
        action="create",
        critic_score=5,
        hard_gates_pass=True,
    )
    base.update(kw)
    return AtomicNoteDraft(**base)


def _citation() -> CitationMeta:
    return CitationMeta(author="Autor", year="2020", title=None, doi=None, source_file="x.pdf")


def _claim(text: str = "Claim-Text (S. 1).") -> Claim:
    return Claim(text=text, anchor_page=1, anchor_span=(0, len(text)), risk_types=["number"], is_quote=False)


# ---- Routing-Veto (auto_write_decision) -------------------------------------


def test_auto_write_blocks_when_faithfulness_fail():
    note = _draft(faithfulness_fail=True)
    auto, reason = auto_write_decision(note)
    assert auto is False
    assert "Faithfulness" in reason


def test_faithfulness_veto_overrides_strong_hub():
    # Sonst voller Hub-Ausnahme-Pfad (action=hub, Score>=4, >=2 Sub-Konzepte,
    # hard_gates_pass=False) — das Veto muss trotzdem greifen.
    note = _draft(
        action="hub",
        critic_score=4,
        hard_gates_pass=False,
        hub_subconcepts=["A", "B"],
        faithfulness_fail=True,
    )
    auto, reason = auto_write_decision(note)
    assert auto is False
    assert "Faithfulness" in reason


def test_auto_write_unaffected_when_faithfulness_pass():
    note = _draft(faithfulness_fail=False)
    assert auto_write_decision(note) == (True, "ok")


# ---- Gate-Aufruf (_apply_faithfulness_gate) ---------------------------------


def test_apply_faithfulness_gate_sets_fail_and_flag(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_FAITHFULNESS_GATE", True)

    verdict = ClaimVerdict(claim=_claim(), status="failed_entailment", evidence="Beleg-Satz", entailment=0.1)
    fake_gate = GateResult(verdicts=[verdict], failed=True, n_supported=0, n_failed=1, n_abstained=0)
    monkeypatch.setattr(
        "generative.pipeline.faithfulness_gate.run_faithfulness_gate",
        lambda body, page_index, citation, **kw: fake_gate,
    )

    draft = _draft(action="create")
    orchestrator._apply_faithfulness_gate(draft, {1: "text"}, _citation())

    assert draft.faithfulness_fail is True
    assert any("Faithfulness" in f for f in draft.quality_flags)


def test_apply_faithfulness_gate_skipped_when_action_not_create(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_FAITHFULNESS_GATE", True)
    called: list[bool] = []

    def _fake_run_gate(*a, **kw):
        called.append(True)
        return GateResult(verdicts=[], failed=True, n_supported=0, n_failed=1, n_abstained=0)

    monkeypatch.setattr("generative.pipeline.faithfulness_gate.run_faithfulness_gate", _fake_run_gate)

    draft = _draft(action="extend")
    orchestrator._apply_faithfulness_gate(draft, {1: "text"}, _citation())

    assert called == []
    assert draft.faithfulness_fail is False


def test_apply_faithfulness_gate_skipped_when_no_page_index(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_FAITHFULNESS_GATE", True)
    called: list[bool] = []

    def _fake_run_gate(*a, **kw):
        called.append(True)
        return GateResult(verdicts=[], failed=True, n_supported=0, n_failed=1, n_abstained=0)

    monkeypatch.setattr("generative.pipeline.faithfulness_gate.run_faithfulness_gate", _fake_run_gate)

    draft = _draft(action="create")
    orchestrator._apply_faithfulness_gate(draft, None, _citation())

    assert called == []
    assert draft.faithfulness_fail is False


def test_apply_faithfulness_gate_skipped_when_empty_page_index(monkeypatch):
    # build_page_index("") liefert {} (markerloser Text) — auch das leere Dict
    # muss das Gate skippen, nicht nur None (Mistral-Review E6, LOW).
    monkeypatch.setattr(orchestrator, "ENABLE_FAITHFULNESS_GATE", True)
    called: list[bool] = []

    def _fake_run_gate(*a, **kw):
        called.append(True)
        return GateResult(verdicts=[], failed=True, n_supported=0, n_failed=1, n_abstained=0)

    monkeypatch.setattr("generative.pipeline.faithfulness_gate.run_faithfulness_gate", _fake_run_gate)

    draft = _draft(action="create")
    orchestrator._apply_faithfulness_gate(draft, {}, _citation())

    assert called == []
    assert draft.faithfulness_fail is False


def test_apply_faithfulness_gate_abstain_only_no_fail(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_FAITHFULNESS_GATE", True)

    verdict = ClaimVerdict(claim=_claim(), status="abstain_no_window", evidence=None, entailment=None)
    fake_gate = GateResult(verdicts=[verdict], failed=False, n_supported=0, n_failed=0, n_abstained=2)
    monkeypatch.setattr(
        "generative.pipeline.faithfulness_gate.run_faithfulness_gate",
        lambda body, page_index, citation, **kw: fake_gate,
    )

    draft = _draft(action="create")
    orchestrator._apply_faithfulness_gate(draft, {1: "text"}, _citation())

    assert draft.faithfulness_fail is False
    assert any("abstain" in f.lower() for f in draft.quality_flags)
